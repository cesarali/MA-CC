# Check the University of Potsdam LLM proxy through the Windows network stack.
# The Python launcher invokes this helper automatically when running under WSL.

param(
    [string]$Model = "",
    [double]$TimeoutSec = 30,
    [switch]$Chat
)

$ErrorActionPreference = "Stop"
$DefaultModel = "gwdg/qwen3-30b-a3b-instruct-2507"

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
    throw "Could not find the repository-root .env file."
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

function Get-HttpStatusCode {
    param([System.Management.Automation.ErrorRecord]$ErrorRecord)

    if ($null -ne $ErrorRecord.Exception.Response -and
        $null -ne $ErrorRecord.Exception.Response.StatusCode) {
        return [int]$ErrorRecord.Exception.Response.StatusCode
    }
    return $null
}

try {
    $envFile = Find-EnvFile -StartDirectory $PSScriptRoot
    $apiKey = Read-EnvValue -Path $envFile -Name "POTSDAM_API_KEY"
    $baseUrl = Read-EnvValue -Path $envFile -Name "BASE_POTSDAM_LLM_URL"
    $envModel = Read-EnvValue -Path $envFile -Name "POTSDAM_MODEL"

    if (-not $apiKey) {
        throw "POTSDAM_API_KEY is missing or empty in .env."
    }
    if (-not $baseUrl) {
        throw "BASE_POTSDAM_LLM_URL is missing or empty in .env."
    }
    if (-not $Model) {
        $Model = if ($envModel) { $envModel } else { $DefaultModel }
    }

    $baseUrl = $baseUrl.TrimEnd("/")
    $headers = @{ Authorization = "Bearer $apiKey" }
    Write-Output "Configuration loaded from $envFile; checking model '$Model' via Windows."

    $availableModels = $null
    $apiPrefix = $null
    foreach ($prefix in @("", "/v1")) {
        $endpoint = "$baseUrl$prefix/models"
        try {
            $response = Invoke-RestMethod `
                -Method Get `
                -Uri $endpoint `
                -Headers $headers `
                -TimeoutSec $TimeoutSec
            $availableModels = @(
                $response.data |
                    Where-Object { $null -ne $_.id -and [string]$_.id } |
                    ForEach-Object { [string]$_.id } |
                    Sort-Object -Unique
            )
            $apiPrefix = $prefix
            break
        }
        catch {
            $statusCode = Get-HttpStatusCode -ErrorRecord $_
            if ($statusCode -eq 404 -and -not $prefix) {
                continue
            }
            throw
        }
    }

    if ($null -eq $apiPrefix -or $availableModels.Count -eq 0) {
        throw "The model-list request succeeded but returned no model IDs."
    }
    if ($Model -notin $availableModels) {
        throw "Model '$Model' is not currently listed by the University LLM proxy."
    }

    Write-Output "University proxy reachable; $($availableModels.Count) models listed."
    Write-Output "Requested model available: True ($Model)"

    if (-not $Chat) {
        Write-Output "Connectivity/model check passed; use --chat for a completion test."
        exit 0
    }

    Write-Output "Starting minimal chat test with $Model ..."
    $payload = @{
        model = $Model
        messages = @(
            @{ role = "user"; content = "Reply exactly: API works." }
        )
        temperature = 0
        max_tokens = 16
    } | ConvertTo-Json -Depth 5
    $chatResponse = Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUrl$apiPrefix/chat/completions" `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $payload `
        -TimeoutSec $TimeoutSec

    $content = [string]$chatResponse.choices[0].message.content
    Write-Output "Chat result received: '$($content.Trim())'"
    if ($null -ne $chatResponse.usage) {
        Write-Output "Usage: $($chatResponse.usage | ConvertTo-Json -Compress)"
    }
}
catch {
    $statusCode = Get-HttpStatusCode -ErrorRecord $_
    if ($null -ne $statusCode) {
        [Console]::Error.WriteLine("API check failed (HTTP $statusCode).")
    }
    elseif ($_.Exception.Message -match "missing|empty|not currently listed|returned no model") {
        [Console]::Error.WriteLine("API check failed: $($_.Exception.Message)")
    }
    else {
        [Console]::Error.WriteLine(
            "API check failed before receiving an HTTP response. Confirm that " +
            "Cisco AnyConnect is connected in Windows and has installed a route " +
            "to the University network."
        )
    }
    exit 2
}
