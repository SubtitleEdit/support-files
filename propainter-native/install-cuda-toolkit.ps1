param([Parameter(Mandatory=$true)][string] $Destination)
$ErrorActionPreference = 'Stop'
$baseUrl = 'https://developer.download.nvidia.com/compute/cuda/redist'
$components = @(
    @{ Name = 'cuda_nvcc'; Version = '12.8.93'; Sha256 = '9fdc70b4271ed9aad4d64cd7076a7d96ec36512d074b9995fe638de669197391' }
    @{ Name = 'cuda_cudart'; Version = '12.8.90'; Sha256 = '4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b' }
    @{ Name = 'cuda_cccl'; Version = '12.8.90'; Sha256 = 'bd8548fa1ae82f92910bebc3079e14bd58c5a92aa64596d46bd610a478cb39d7' }
    @{ Name = 'libcublas'; Version = '12.8.4.1'; Sha256 = '57a470112cec7e112c95253dde8b3c7184d795dbd92b0bde77a4cb7f8c94c8aa' }
)
$destination = (New-Item -ItemType Directory -Path $Destination -Force).FullName
$licenses = (New-Item -ItemType Directory -Path (Join-Path $destination 'licenses') -Force).FullName
foreach ($component in $components) {
    $stem = "$($component.Name)-windows-x86_64-$($component.Version)-archive"
    $scratch = (New-Item -ItemType Directory -Path (Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString()))).FullName
    try {
        $zip = Join-Path $scratch "$stem.zip"
        curl.exe --fail --location --retry 3 --silent --show-error --output $zip "$baseUrl/$($component.Name)/windows-x86_64/$stem.zip"
        if ($LASTEXITCODE -ne 0) { throw "Could not download $stem." }
        if ((Get-FileHash -Algorithm SHA256 $zip).Hash -ne $component.Sha256) { throw "Unexpected SHA-256 for $stem." }
        tar.exe -xf $zip -C $scratch
        if ($LASTEXITCODE -ne 0) { throw "Could not extract $stem." }
        $source = Join-Path $scratch $stem
        foreach ($folder in Get-ChildItem -Directory -LiteralPath $source) {
            $target = (New-Item -ItemType Directory -Path (Join-Path $destination $folder.Name) -Force).FullName
            Get-ChildItem -LiteralPath $folder.FullName | Copy-Item -Destination $target -Recurse -Force
        }
        if ($component.Name -in 'cuda_cudart', 'libcublas') {
            $notice = Get-ChildItem -LiteralPath $source -File | Where-Object { $_.Name -in 'LICENSE', 'LICENSE.txt' } | Select-Object -First 1
            if (!$notice) { throw "Missing NVIDIA license in $stem." }
            Copy-Item $notice.FullName (Join-Path $licenses "LICENSE-$($component.Name).txt")
        }
    } finally {
        Remove-Item -LiteralPath $scratch -Recurse -Force
    }
}
