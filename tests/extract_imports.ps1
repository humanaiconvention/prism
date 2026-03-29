$files = Get-ChildItem "D:\Genesis\tests\test_phase*.py"
$imports = @()
foreach ($f in $files) {
    if ($f.Name -match "test_phase([1-2][1-9]|[2][0-4])") {
        $content = Get-Content $f.FullName -Raw
        $matches = [regex]::Matches($content, 'from\s+scripts.run_phase[a-zA-Z0-9_]+\s+import\s+(?:\(([^)]+)\)|([^\n]+))')
        foreach ($m in $matches) {
            $names = if ($m.Groups[1].Success) { $m.Groups[1].Value } else { $m.Groups[2].Value }
            $names = $names -replace '\s+', '' -split ',' | Where-Object { $_ -ne '' }
            $imports += $names
        }
    }
}
$imports | Sort-Object -Unique | Out-File D:\prism\tests\imported_funcs.txt
