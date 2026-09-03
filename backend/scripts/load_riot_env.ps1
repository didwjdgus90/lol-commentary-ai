$ErrorActionPreference = "Stop"

$repositoryRoot = (
    Resolve-Path (
        Join-Path `
          $PSScriptRoot `
          "..\.."
    )
).Path

$envFile = Join-Path `
  $repositoryRoot `
  ".env"

if (-not (Test-Path $envFile)) {
    throw "Local .env file was not found."
}

$riotLines = @(
    Get-Content $envFile |
    Where-Object {
        $_.Trim() -match '^RIOT_API_KEY='
    }
)

if ($riotLines.Count -ne 1) {
    throw (
        "RIOT_API_KEY must exist " +
        "exactly once in .env"
    )
}

$riotKey = (
    $riotLines[0].Split(
        "=",
        2
    )[1]
).Trim()

if (
    [string]::IsNullOrWhiteSpace(
        $riotKey
    )
) {
    throw "RIOT_API_KEY is empty"
}

if ($riotKey -eq "change-me") {
    throw (
        "RIOT_API_KEY is still " +
        "change-me"
    )
}

[Environment]::SetEnvironmentVariable(
    "RIOT_API_KEY",
    $riotKey,
    "Process"
)

$riotKey = $null
$riotLines = $null

Write-Host (
    "Riot API environment loaded."
)

Write-Host (
    "RIOT_API_KEY_SET=True"
)