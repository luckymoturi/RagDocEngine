@echo off
echo ========================================
echo ChromaDB Force Reset
echo ========================================
echo.
echo WARNING: This will delete your database!
echo You will need to re-upload all documents.
echo.
pause

echo.
echo Stopping Python processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo Deleting old database...
if exist chroma_db (
    rmdir /s /q chroma_db
    echo Done!
) else (
    echo No database found.
)

echo Creating new database folder...
mkdir chroma_db

echo.
echo ========================================
echo SUCCESS!
echo ========================================
echo.
echo Next steps:
echo 1. Start server: python main.py
echo 2. Re-upload your PDF documents
echo.
pause
