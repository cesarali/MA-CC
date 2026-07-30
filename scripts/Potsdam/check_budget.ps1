# Query the University of Potsdam LLM budget through the Windows network stack.
# From WSL, run:
# powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w scripts/Potsdam/check_budget.ps1)"

param(
    [double]$TimeoutSec = 30
)

$ErrorActionPreference = "Stop"

function Find-EnvFile {
    param([string]$StartDirectory)

    $directory = [System.IO.DirectoryInfo]::new($StartDirectory)
    while ($null -ne $directory) {
        $candidate = Join-Path $directory.FullName ".env"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
        $directory = $directory.Parent
    }
    throw "Could not find a .env file in the script directory or its parents."
}

function Read-EnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        if ($parts.Count -eq 2 -and $parts[0].Trim() -eq $Name) {
            $value = $parts[1].Trim()
            if ($value.Length -ge 2) {
                $quoted = ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                    ($value.StartsWith("'") -and $value.EndsWith("'"))
                if ($quoted) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
            }
            return $value
        }
    }
    return $null
}

function Find-ResponseValue {
    param(
        [object]$InputObject,
        [string]$Name
    )

    if ($null -eq $InputObject) {
        return $null
    }

    if ($InputObject -is [System.Collections.IDictionary]) {
        if ($InputObject.Contains($Name) -and $null -ne $InputObject[$Name]) {
            return $InputObject[$Name]
        }
        foreach ($value in $InputObject.Values) {
            $found = Find-ResponseValue -InputObject $value -Name $Name
            if ($null -ne $found) {
                return $found
            }
        }
    }
    elseif ($InputObject -is [pscustomobject]) {
        $property = $InputObject.PSObject.Properties[$Name]
        if ($null -ne $property -and $null -ne $property.Value) {
            return $property.Value
        }
        foreach ($child in $InputObject.PSObject.Properties.Value) {
            $found = Find-ResponseValue -InputObject $child -Name $Name
            if ($null -ne $found) {
                return $found
            }
        }
    }
    elseif ($InputObject -is [System.Collections.IEnumerable] -and
            $InputObject -isnot [string]) {
        foreach ($child in $InputObject) {
            $found = Find-ResponseValue -InputObject $child -Name $Name
            if ($null -ne $found) {
                return $found
            }
        }
    }
    return $null
}

function Format-Number {
    param([object]$Value)

    $number = 0.0
    if ([double]::TryParse(
        [string]$Value,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$number
    )) {
        return $number.ToString("0.######", [System.Globalization.CultureInfo]::InvariantCulture)
    }
    return [string]$Value
}

try {
    $envFile = Find-EnvFile -StartDirectory $PSScriptRoot
    $apiKey = Read-EnvValue -Path $envFile -Name "POTSDAM_API_KEY"
    $baseUrl = Read-EnvValue -Path $envFile -Name "BASE_POTSDAM_LLM_URL"
    if (-not $apiKey -or -not $baseUrl) {
        throw "POTSDAM_API_KEY and BASE_POTSDAM_LLM_URL must be set in .env."
    }

    $endpoint = $baseUrl.TrimEnd("/") + "/user/info"
    $response = Invoke-RestMethod `
        -Method Get `
        -Uri $endpoint `
        -Headers @{ Authorization = "Bearer $apiKey" } `
        -TimeoutSec $TimeoutSec

    $budget = Find-ResponseValue -InputObject $response -Name "max_budget"
    if ($null -eq $budget) {
        $budget = Find-ResponseValue -InputObject $response -Name "budget"
    }
    $spend = Find-ResponseValue -InputObject $response -Name "spend"

    Write-Output "University of Potsdam LLM account"
    if ($null -ne $budget) {
        Write-Output "Budget:    $(Format-Number $budget)"
    }
    if ($null -ne $spend) {
        Write-Output "Spent:     $(Format-Number $spend)"
    }

    $budgetNumber = 0.0
    $spendNumber = 0.0
    $budgetIsNumber = [double]::TryParse(
        [string]$budget,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$budgetNumber
    )
    $spendIsNumber = [double]::TryParse(
        [string]$spend,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$spendNumber
    )
    if ($budgetIsNumber -and $spendIsNumber) {
        Write-Output "Remaining: $(Format-Number ($budgetNumber - $spendNumber))"
        if ($budgetNumber -gt 0) {
            Write-Output ("Used:      {0:0.00}%" -f (($spendNumber / $budgetNumber) * 100))
        }
    }

    $extraFields = [ordered]@{
        soft_budget = "Soft budget"
        budget_duration = "Budget period"
        budget_reset_at = "Budget reset"
        rpm_limit = "RPM limit"
        tpm_limit = "TPM limit"
    }
    foreach ($field in $extraFields.Keys) {
        $value = Find-ResponseValue -InputObject $response -Name $field
        if ($null -ne $value) {
            Write-Output "$($extraFields[$field]): $(Format-Number $value)"
        }
    }

    if ($null -eq $budget -and $null -eq $spend) {
        Write-Error "The request succeeded, but no recognized budget or spend fields were returned."
        exit 3
    }
}
catch {
    $statusCode = $null
    if ($null -ne $_.Exception.Response -and
        $null -ne $_.Exception.Response.StatusCode) {
        $statusCode = [int]$_.Exception.Response.StatusCode
    }
    if ($null -ne $statusCode) {
        Write-Error "Budget request failed (HTTP $statusCode)."
    }
    else {
        Write-Error "Budget request failed. Confirm that the VPN is connected in Windows."
    }
    exit 1
}
