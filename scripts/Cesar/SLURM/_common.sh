#!/usr/bin/env bash

# Shared runtime checks for generic Cesar MA-CC jobs.
#
# Cesar is the private SLURM-on-Kubernetes cluster reached through Tailscale at
# slurm-login.tail769bd2.ts.net. It differs from the other sites in three ways
# that this file exists to absorb:
#
#   1. There is no Conda and no module system. The Python environment is a uv
#      virtualenv on the shared NFS mount, visible to every compute node.
#   2. The compute node image ships no CA certificates, so TLS verification
#      fails for every HTTPS call unless SSL_CERT_FILE points at a bundle we
#      stage ourselves. Without this, every provider request fails.
#   3. There is no cgroup confinement (TaskPlugin is unset), so `nproc` reports
#      the 16 physical host cores even though SLURM allocates 4 CPUs. BLAS would
#      otherwise start 16 threads per task and thrash when tasks share a node.

readonly CESAR_DEFAULT_SHARED_ROOT=/shared
readonly CESAR_DEFAULT_MIN_SHARED_KB=1048576

if [[ -n "${CESAR_REPO_ROOT:-}" ]]; then
  CESAR_REPO_ROOT="$(cd -- "${CESAR_REPO_ROOT}" && pwd -L)"
  CESAR_SCRIPT_DIR="${CESAR_REPO_ROOT}/scripts/Cesar/SLURM"
else
  CESAR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -L)"
  CESAR_REPO_ROOT="$(cd -- "${CESAR_SCRIPT_DIR}/../../.." && pwd -L)"
fi
readonly CESAR_SCRIPT_DIR CESAR_REPO_ROOT

cesar_die() {
  echo "error: $*" >&2
  exit 2
}

# Resolve the interpreter of the shared uv virtualenv. There is no Conda here,
# so this returns a python3 path rather than a conda executable.
cesar_resolve_python() {
  if [[ -n "${CESAR_PYTHON:-}" ]]; then
    [[ -x "${CESAR_PYTHON}" ]] || \
      cesar_die "CESAR_PYTHON is not executable: ${CESAR_PYTHON}"
    printf '%s\n' "${CESAR_PYTHON}"
    return
  fi
  local candidate
  for candidate in \
    "${CESAR_ENV_ROOT:-${HOME}/envs/MA-CC}/bin/python" \
    /shared/home/"${USER}"/envs/MA-CC/bin/python
  do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  cesar_die "no MA-CC environment found; create one with uv under \${HOME}/envs/MA-CC"
}

# Stage the CA bundle into the environment. The compute node image has an empty
# /etc/ssl/certs, so this is mandatory rather than defensive: every DeepInfra
# request fails with CERTIFICATE_VERIFY_FAILED without it.
cesar_require_ca_bundle() {
  local bundle="${CESAR_CA_BUNDLE:-${HOME}/etc/ca-certificates.crt}"
  if [[ ! -r "${bundle}" ]]; then
    cesar_die "CA bundle missing or unreadable: ${bundle}
  Compute nodes ship no certificates. Stage one on the shared mount with:
    mkdir -p \"\${HOME}/etc\"
    cp /etc/ssl/certs/ca-certificates.crt \"\${HOME}/etc/ca-certificates.crt\"
  (run that on the login node, which does have certificates)"
  fi
  export SSL_CERT_FILE="${bundle}"
  export REQUESTS_CA_BUNDLE="${bundle}"
  export CURL_CA_BUNDLE="${bundle}"
}

# SLURM allocates CPUs but does not confine them here, so pin the numeric
# thread pools to the allocation instead of letting them read the host.
cesar_limit_thread_pools() {
  local allocated="${SLURM_CPUS_PER_TASK:-1}"
  [[ "${allocated}" =~ ^[1-9][0-9]*$ ]] || allocated=1
  export OMP_NUM_THREADS="${allocated}"
  export OPENBLAS_NUM_THREADS="${allocated}"
  export MKL_NUM_THREADS="${allocated}"
  export NUMEXPR_NUM_THREADS="${allocated}"
}

cesar_prepare_runtime() {
  local shared_root="${CESAR_SHARED_ROOT:-${CESAR_DEFAULT_SHARED_ROOT}}"
  local minimum_kb="${CESAR_MIN_SHARED_KB:-${CESAR_DEFAULT_MIN_SHARED_KB}}"
  [[ "${shared_root}" == /shared || "${shared_root}" == /shared/* ]] || \
    cesar_die "Cesar caches must stay under /shared"
  [[ "${minimum_kb}" =~ ^[1-9][0-9]*$ ]] || \
    cesar_die "CESAR_MIN_SHARED_KB must be a positive integer"
  mountpoint -q /shared || \
    cesar_die "/shared is not mounted on $(hostname); the study cannot read its manifest"
  local available_kb
  available_kb="$(df -Pk "${shared_root}" | awk 'NR == 2 {print $4}')"
  [[ "${available_kb}" =~ ^[0-9]+$ ]] || \
    cesar_die "could not determine free space on ${shared_root}"
  (( available_kb >= minimum_kb )) || \
    cesar_die "${shared_root} has less than ${minimum_kb} KiB available"

  cesar_require_ca_bundle
  cesar_limit_thread_pools

  export XDG_CACHE_HOME="${shared_root}/cache/${USER}/xdg"
  export COMET_CACHE_DIR="${shared_root}/cache/${USER}/comet"
  export MPLCONFIGDIR="${shared_root}/cache/${USER}/matplotlib"
  mkdir -p "${XDG_CACHE_HOME}" "${COMET_CACHE_DIR}" "${MPLCONFIGDIR}"
  cd "${CESAR_REPO_ROOT}"
}
