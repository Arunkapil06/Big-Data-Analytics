$hadoopEnvPath = "C:\hadoop-3.3.6\etc\hadoop\hadoop-env.cmd"
$javaHome = "C:\Program Files\Java\jdk1.8.0_202"
$javaHomePath = $javaHome -replace '\\', '\\'

# Read the content of the file
$content = Get-Content $hadoopEnvPath -Raw

# Replace the JAVA_HOME line with escaped backslashes
$newContent = $content -replace 'set JAVA_HOME=.*', "set JAVA_HOME=$javaHomePath"

# Write the content back to the file
$newContent | Set-Content $hadoopEnvPath -Force 