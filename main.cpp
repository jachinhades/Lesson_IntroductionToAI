
#include <bits/stdc++.h>

using namespace std;

using Image = struct {
    int rows, cols;
    vector<vector<float>> pixels;
};

using ConvWeights = vector<vector<vector<vector<float>>>>;
using Biases = vector<float>;

vector<Image> read_mnist_images(const string& filename, int num_images) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "无法打开图片文件：" << filename << endl;
        return {};
    }

    int magic_number = 0, total_images = 0, rows = 0, cols = 0;
    file.read((char*)&magic_number, 4);
    file.read((char*)&total_images, 4);
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);

    magic_number = __builtin_bswap32(magic_number);
    total_images = __builtin_bswap32(total_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    vector<Image> images;
    for (int i = 0; i < num_images && i < total_images; ++i) {
        Image img;
        img.rows = rows;
        img.cols = cols;
        img.pixels.resize(rows, vector<float>(cols));

        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, 1);
                img.pixels[r][c] = pixel / 255.0f;
            }

        images.push_back(img);
    }

    return images;
}

ConvWeights initialize_conv_weights(int out_ch, int in_ch, int kernel_size) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0, 0.1);

    ConvWeights weights(out_ch, vector<vector<vector<float>>>(
        in_ch, vector<vector<float>>(kernel_size, vector<float>(kernel_size))));

    for (int oc = 0; oc < out_ch; ++oc)
        for (int ic = 0; ic < in_ch; ++ic)
            for (int i = 0; i < kernel_size; ++i)
                for (int j = 0; j < kernel_size; ++j)
                    weights[oc][ic][i][j] = dist(gen);

    return weights;
}

vector<vector<vector<float>>> conv_forward(
    const vector<vector<float>>& input,
    const ConvWeights& weights,
    const Biases& biases,
    int stride = 1)
{
    int input_h = input.size();
    int input_w = input[0].size();
    int out_ch = weights.size();
    int kernel_size = weights[0][0].size();
    int output_h = (input_h - kernel_size) / stride + 1;
    int output_w = (input_w - kernel_size) / stride + 1;

    vector<vector<vector<float>>> output(out_ch,
        vector<vector<float>>(output_h, vector<float>(output_w, 0.0f)));

    for (int oc = 0; oc < out_ch; ++oc) {
        for (int i = 0; i < output_h; ++i) {
            for (int j = 0; j < output_w; ++j) {
                float sum = biases[oc];
                for (int ic = 0; ic < weights[0].size(); ++ic) {
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int ih = i * stride + ki;
                            int iw = j * stride + kj;
                            sum += input[ih][iw] * weights[oc][ic][ki][kj];
                        }
                    }
                }
                output[oc][i][j] = sum;
            }
        }
    }

    return output;
}

int main() {
    system("chcp 65001");
    string images_path = "train-images-idx3-ubyte";
    vector<Image> images = read_mnist_images(images_path, 1);
    if (images.empty()) {
        cerr << "读取图片失败。" << endl;
        return 1;
    }
    cout << "成功读取图像！尺寸为：" << images[0].rows << "x" << images[0].cols << endl;

    cout << "\n图像像素值：\n";
    for (int i = 0; i < images[0].rows; ++i) {
        for (int j = 0; j < images[0].cols; ++j)
            cout << images[0].pixels[i][j] << " ";
        cout << endl;
    }

    ConvWeights conv_weights = initialize_conv_weights(8, 1, 3);
    Biases conv_biases(8, 0.0f);

    cout << "\n卷积核[0][0]内容：\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            cout << conv_weights[0][0][i][j] << " ";
        cout << endl;
    }

    auto output = conv_forward(images[0].pixels, conv_weights, conv_biases);

    cout << "\n卷积输出尺寸: " << output.size() << " x " << output[0].size() << " x " << output[0][0].size() << endl;
    cout << "第一个通道完整输出：\n";
    for (int i = 0; i < output[0].size(); ++i) {
        for (int j = 0; j < output[0][0].size(); ++j) {
            cout << output[0][i][j] << " ";
        }
        cout << endl;
    }
    ofstream fout("conv_output.txt");
    for (int i = 0; i < output[0].size(); ++i) {
        for (int j = 0; j < output[0][0].size(); ++j)
            fout << output[0][i][j] << " ";
        fout << "\n";
    }
    fout.close();
    return 0;
}
