param([Parameter(Mandatory=$true)][string] $Executable)
$ErrorActionPreference = "Stop"
$dir = New-Item -ItemType Directory -Path ([IO.Path]::Combine([IO.Path]::GetTempPath(), [Guid]::NewGuid()))
try {
    $width = 32
    $height = 24
    $cells = $width * $height
    $rgb = [byte[]]::new(3 * $cells)
    $mask = [byte[]]::new($cells)
    for ($y = 0; $y -lt $height; $y++) {
        for ($x = 0; $x -lt $width; $x++) {
            $p = $y * $width + $x
            $subtitle = $x -ge 9 -and $x -lt 22 -and $y -ge 15 -and $y -lt 18
            if ($subtitle) { $mask[$p] = 255 }
            for ($c = 0; $c -lt 3; $c++) {
                $rgb[3 * $p + $c] = if ($subtitle) { 255 } else { 70 + 2 * $x + 2 * $y + 20 * $c }
            }
        }
    }
    $stream = [IO.File]::Create((Join-Path $dir.FullName "stream.bin"))
    try {
        for ($i = 0; $i -lt 2; $i++) {
            $stream.Write($rgb, 0, $rgb.Length)
            $stream.Write($mask, 0, $mask.Length)
        }
    } finally {
        $stream.Dispose()
    }
    $output = Join-Path $dir.FullName "out.rgb"
    $start = @{
        FilePath = $Executable
        ArgumentList = @("--stream", "$($width)x$($height)")
        NoNewWindow = $true
        Wait = $true
        PassThru = $true
        RedirectStandardInput = Join-Path $dir.FullName "stream.bin"
        RedirectStandardOutput = $output
    }
    $process = Start-Process @start
    if ($process.ExitCode -ne 0) { throw "Native stream failed with exit code $($process.ExitCode)." }
    $bytes = [IO.File]::ReadAllBytes($output)
    if ($bytes.Length -ne 2 * $rgb.Length) { throw "Unexpected number of output bytes." }
    $changed = 0
    for ($i = 0; $i -lt $bytes.Length; $i++) {
        $j = $i % $rgb.Length
        if ($bytes[$i] -ne $rgb[$j]) {
            if ($mask[[int][Math]::Floor($j / 3)] -eq 0) { throw "Unmasked pixel changed." }
            $changed++
        }
    }
    if ($changed -eq 0) { throw "Masked pixels were not inpainted." }
    Write-Host "Rendered two frames and preserved all pixels outside the masks."
} finally {
    Remove-Item -LiteralPath $dir.FullName -Recurse -Force
}
