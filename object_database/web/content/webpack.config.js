const path = require('path');

module.exports = {
    devtool: 'inline-source-map',
    mode: 'development',
    entry: './main.js',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'main.bundle.js'
    }
};
