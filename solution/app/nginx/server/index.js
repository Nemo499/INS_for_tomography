const express = require('express');
const funcs = require('./funcs.js')
const multer = require('multer');

const PORT = 3330;
const HOST = '127.0.0.1';

const app = express();
app.use(express.json());

const upload = multer({ dest: 'uploads/' });
app.post('/api/upload', upload.single('archive'), (req, res) => {
    console.log('Received request to /api/upload');
    if (!req.file) {
        return res.status(400).send('No files were uploaded.');
    }
    res.status(200).send('File uploaded successfully');
});

app.listen(PORT, () => {
  console.log(`Сервер запущен на http://${HOST}:${PORT}`);
}); 

