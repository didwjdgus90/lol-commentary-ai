$ErrorActionPreference = "Stop"

$repositoryRoot = (
    Resolve-Path (
        Join-Path $PSScriptRoot "..\.."
    )
).Path

$envFile = Join-Path $repositoryRoot ".env"

if (-not (Test-Path $envFile)) {
    throw @"
Local environment file was not found:

$envFile

Create it from:

.env.example
"@
}

$values = @{}

foreach ($rawLine in Get-Content $envFile) {
    $line = $rawLine.Trim()

    if (
        [string]::IsNullOrWhiteSpace($line) -or
        $line.StartsWith("#")
    ) {
        continue
    }

    $parts = $line.Split("=", 2)

    if ($parts.Count -ne 2) {
        throw "Invalid .env line: $line"
    }

    $key = $parts[0].Trim()
    $value = $parts[1].Trim()

    if ([string]::IsNullOrWhiteSpace($key)) {
        throw "Environment variable name is empty."
    }

    $values[$key] = $value
}

$required = @(
    "LOL_POSTGRES_HOST",
    "LOL_POSTGRES_PORT",
    "LOL_POSTGRES_DB",
    "LOL_POSTGRES_USER",
    "LOL_POSTGRES_PASSWORD"
)

foreach ($key in $required) {
    if (
        -not $values.ContainsKey($key) -or
        [string]::IsNullOrWhiteSpace($values[$key])
    ) {
        throw "Missing required environment value: $key"
    }
}

if ($values["LOL_POSTGRES_PASSWORD"] -eq "change-me") {
    throw @"
LOL_POSTGRES_PASSWORD is still 'change-me'.

Set a local development password in .env first.
"@
}

$port = 0

if (
    -not [int]::TryParse(
        $values["LOL_POSTGRES_PORT"],
        [ref]$port
    )
) {
    throw "LOL_POSTGRES_PORT must be an integer."
}

if (
    $port -lt 1 -or
    $port -gt 65535
) {
    throw "LOL_POSTGRES_PORT must be between 1 and 65535."
}

foreach ($key in $required) {
    [Environment]::SetEnvironmentVariable(
        $key,
        $values[$key],
        "Process"
    )
}

$user = $values["LOL_POSTGRES_USER"]
$password = $values["LOL_POSTGRES_PASSWORD"]
$hostName = $values["LOL_POSTGRES_HOST"]
$database = $values["LOL_POSTGRES_DB"]

$escapedUser = [Uri]::EscapeDataString($user)
$escapedPassword = [Uri]::EscapeDataString($password)
$escapedDatabase = [Uri]::EscapeDataString($database)

$databaseUrl = (
    "postgresql://" +
    $escapedUser +
    ":" +
    $escapedPassword +
    "@" +
    $hostName +
    ":" +
    $port +
    "/" +
    $escapedDatabase
)

[Environment]::SetEnvironmentVariable(
    "LOL_DATABASE_URL",
    $databaseUrl,
    "Process"
)

Write-Host "LoL Commentary AI development environment loaded."
Write-Host "PostgreSQL host: $hostName"
Write-Host "PostgreSQL port: $port"
Write-Host "PostgreSQL database: $database"
Write-Host "PostgreSQL user: $user"
Write-Host "LOL_DATABASE_URL_SET=True"