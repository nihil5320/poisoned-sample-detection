function Rename-Files {
    param (
        [string]$FolderPath,
        [string]$OldString,
        [string]$NewString
    )

    if (-not (Test-Path $FolderPath)) {
        Write-Host "Folder path '$FolderPath' does not exist."
        return
    }

    $files = Get-ChildItem -Path $FolderPath

    foreach ($file in $files) {
        $newName = $file.Name -replace $OldString, $NewString
        $newPath = Join-Path -Path $FolderPath -ChildPath $newName

        if ($file.FullName -ne $newPath) {
            Rename-Item -Path $file.FullName -NewName $newName
            Write-Host "Renamed $($file.Name) to $newName"
        }
    }
}

function Move-RandomFiles {
    param (
        [string]$SourceFolder,
        [string]$DestinationFolder,
        [int]$NumberOfFiles
    )

    if (-not (Test-Path $SourceFolder)) {
        Write-Host "Source folder '$SourceFolder' does not exist."
        return
    }

    if (-not (Test-Path $DestinationFolder)) {
        Write-Host "Destination folder '$DestinationFolder' does not exist."
        return
    }

    $files = Get-ChildItem -Path $SourceFolder | Get-Random -Count $NumberOfFiles

    foreach ($file in $files) {
        $destinationPath = Join-Path -Path $DestinationFolder -ChildPath $file.Name
        Move-Item -Path $file.FullName -Destination $destinationPath
        Write-Host "Moved $($file.Name) to $destinationPath"
    }
}

# Rename-Files "C:\Users\Nihil\Desktop\130kv2\train\poisoned" "original" "poisoned"
# Rename-Files "C:\Users\Nihil\Desktop\130kv2\train\poisoned" "-nightshade-intensity-DEFAULT-V1" ""

Move-RandomFiles "C:\Users\Nihil\Desktop\130kv2\train\original" "C:\Users\Nihil\Desktop\130kv2\test\original" 660
Move-RandomFiles "C:\Users\Nihil\Desktop\130kv2\train\original" "C:\Users\Nihil\Desktop\130kv2\validation\original" 660
Move-RandomFiles "C:\Users\Nihil\Desktop\130kv2\train\poisoned" "C:\Users\Nihil\Desktop\130kv2\test\poisoned" 660
Move-RandomFiles "C:\Users\Nihil\Desktop\130kv2\train\poisoned" "C:\Users\Nihil\Desktop\130kv2\validation\poisoned" 660