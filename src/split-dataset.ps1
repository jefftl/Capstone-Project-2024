param(
    [Parameter(Mandatory=$true)]
    [string]$sourceDir = "images",
    [Parameter(Mandatory=$false)]
    [int]$trainPercent = 70,
    [Parameter(Mandatory=$false)]
    [int]$valPercent = 20,
    [Parameter(Mandatory=$false)]
    [int]$testPercent = 10
)

# Validate percentages
if (($trainPercent + $valPercent + $testPercent) -ne 100) {
    throw "Percentages must sum to 100"
}

# Create output directories
$splits = @("train", "val", "test")
foreach ($split in $splits) {
    if (-not (Test-Path "$sourceDir\$split")) {
        New-Item -Path "$sourceDir\$split" -ItemType Directory -Force
    }
}

# Process each class directory
Get-ChildItem $sourceDir -Directory | Where-Object { $_.Name -notin $splits } | ForEach-Object {
    $className = $_.Name
    Write-Host "Processing class: $className"

    # Create class subdirectories in each split
    foreach ($split in $splits) {
        if (-not (Test-Path "$sourceDir\$split\$className")) {
            New-Item -Path "$sourceDir\$split\$className" -ItemType Directory -Force
        }
    }

    # Get all images and shuffle them
    $images = Get-ChildItem "$sourceDir\$className\*.*" | Where-Object { $_.Extension -match '\.(jpg|jpeg|png|gif)$' }
    $shuffledImages = $images | Sort-Object { Get-Random }
    $totalImages = $shuffledImages.Count

    # Calculate split sizes
    $trainSize = [math]::Floor($totalImages * $trainPercent / 100)
    $valSize = [math]::Floor($totalImages * $valPercent / 100)
    $testSize = $totalImages - $trainSize - $valSize

    # Split and copy images
    $currentIndex = 0

    # Training set
    $shuffledImages[0..($trainSize-1)] | ForEach-Object {
        Copy-Item $_.FullName -Destination "$sourceDir\train\$className\"
        Write-Host "Copying $($_.Name) to train set"
    }

    # Validation set
    $shuffledImages[$trainSize..($trainSize+$valSize-1)] | ForEach-Object {
        Copy-Item $_.FullName -Destination "$sourceDir\val\$className\"
        Write-Host "Copying $($_.Name) to validation set"
    }

    # Test set
    $shuffledImages[($trainSize+$valSize)..($totalImages-1)] | ForEach-Object {
        Copy-Item $_.FullName -Destination "$sourceDir\test\$className\"
        Write-Host "Copying $($_.Name) to test set"
    }
}

Write-Host "Dataset splitting complete!"