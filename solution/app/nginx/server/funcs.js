const path = require('path');
const { exec } = require('child_process');
const AdmZip = require('adm-zip')

async function postfile(req, res){
    try {
        const command = `/home/ubuser/OpenFOAM/ubuser-12/run/kurs/web/Run.sh`;

        req.

        exec(command, (error, stdout, stderr) => {
          if (error) {
            console.error(`Ошибка при выполнении расчета: ${error}`);
            return res.status(500).json({ error: 'Ошибка при запуске расчета' });
          }

          unzip()

          res.status(200);
        });

        } catch (error) {
        console.error('Ошибка:', error);
        res.status(500).json({ error: 'Внутренняя ошибка сервера' });
        }
}

async function upload(req, res){
    // Проверка наличия файла в запросе
    if (!req.files || !req.files.archive) {
        return res.status(400).json({ error: 'No archive file uploaded' });
    }

    const archive = req.files.archive;
    // Проверка расширения файла (только ZIP)
    if (archive.mimetype !== 'application/zip') {
        return res.status(400).json({ error: 'Only ZIP archives are supported' });
    }

    // Генерация уникального имени для архива
    const fileName = `${Date.now()}_${archive.name}`;

    savedir = path.join(__dirname, '..', 'services', 'images');
    const filePath = path.join(savedir, fileName);

    // Сохранение архива на сервере
    archive.mv(filePath, (err) => {
        if (err) {
            return res.status(500).json({ error: 'Failed to save archive' });
        }

        try {
            // Распаковка архива
            const zip = new AdmZip(filePath);
            const extractPath = path.join(extractDir, path.basename(fileName, '.zip'));
            zip.extractAllTo(extractPath, true); // true для перезаписи существующих файлов

            // Удаление загруженного архива (опционально)
            fs.unlinkSync(filePath);

            res.json({ message: `Archive ${archive.name} uploaded and extracted to ${extractPath}` });
        } catch (error) {
            res.status(500).json({ error: 'Failed to extract archive' });
        }
    });

}
