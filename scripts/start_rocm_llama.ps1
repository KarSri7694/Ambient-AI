param(
    [Parameter(Mandatory=$true)]
    [string]$LlamaServer,

    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [string]$ApiKey = "testkey",
    [string]$ModelsPreset = ".\models_preset.ini",
    [string]$DefaultModel = "Ambient_Qwen25_VL_4B_ROCm"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LlamaServer -PathType Leaf)) {
    throw "llama-server executable not found: $LlamaServer"
}
if (-not (Test-Path -LiteralPath $ModelsPreset -PathType Leaf)) {
    throw "models preset file not found: $ModelsPreset"
}

& $LlamaServer `
    --host $HostAddress `
    --port $Port `
    --api-key $ApiKey `
    --models-preset $ModelsPreset `
    --model $DefaultModel `
    -ngl 99 `
    -fa on `
    -ctk q8_0 `
    -ctv q8_0

exit $LASTEXITCODE
