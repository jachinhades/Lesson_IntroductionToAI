//
// Created by jachin on 25-4-12.
//
#include <bits/stdc++.h>
using namespace std;

using Image = struct {
    int rows, cols;
    vector<vector<float>> pixels;
};

using ConvWeights = vector<vector<vector<vector<float>>>>; // [out_ch][in_ch][k][k]
using Biases = vector<float>;
using DenseWeights = vector<vector<float>>; // [out_dim][in_dim]

// ---------- 工具函数 ----------
float random_weight(float stddev = 0.1f) {
    static random_device rd;
    static mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, stddev);
    return dist(gen);
}

// ---------- MNIST 读取 ----------
vector<Image> read_mnist_images(const string& filename, int num_images) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) return {};

    int magic = 0, total = 0, rows = 0, cols = 0;
    file.read((char*)&magic, 4);
    file.read((char*)&total, 4);
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);

    magic = __builtin_bswap32(magic);
    total = __builtin_bswap32(total);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    vector<Image> images;
    for (int n = 0; n < num_images && n < total; ++n) {
        Image img{rows, cols, vector<vector<float>>(rows, vector<float>(cols))};
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

vector<uint8_t> read_mnist_labels(const string& filename, int num_labels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) return {};

    int magic = 0, total = 0;
    file.read((char*)&magic, 4);
    file.read((char*)&total, 4);

    magic = __builtin_bswap32(magic);
    total = __builtin_bswap32(total);

    vector<uint8_t> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, 1);
        labels[i] = label;
    }
    return labels;
}

// ---------- 卷积前向 ----------
vector<vector<vector<float>>> conv_forward(const vector<vector<float>>& input,
    const ConvWeights& weights, const Biases& biases, int stride = 1) {

    int H = input.size(), W = input[0].size();
    int out_ch = weights.size(), kernel = weights[0][0].size();
    int out_h = (H - kernel) / stride + 1;
    int out_w = (W - kernel) / stride + 1;

    vector<vector<vector<float>>> output(out_ch,
        vector<vector<float>>(out_h, vector<float>(out_w, 0.0f)));

    for (int oc = 0; oc < out_ch; ++oc)
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j) {
                float sum = biases[oc];
                for (int ic = 0; ic < 1; ++ic)
                    for (int ki = 0; ki < kernel; ++ki)
                        for (int kj = 0; kj < kernel; ++kj)
                            sum += input[i+ki][j+kj] * weights[oc][ic][ki][kj];
                output[oc][i][j] = max(0.0f, sum); // ReLU
            }
    return output;
}

vector<float> flatten(const vector<vector<vector<float>>>& input) {
    vector<float> out;
    for (const auto& ch : input)
        for (const auto& row : ch)
            out.insert(out.end(), row.begin(), row.end());
    return out;
}

vector<float> dense_forward(const vector<float>& input, const DenseWeights& W, const Biases& b) {
    int out_dim = W.size(), in_dim = W[0].size();
    vector<float> out(out_dim, 0.0f);
    for (int i = 0; i < out_dim; ++i) {
        for (int j = 0; j < in_dim; ++j)
            out[i] += W[i][j] * input[j];
        out[i] += b[i];
    }
    return out;
}

vector<float> softmax(const vector<float>& logits) {
    float max_logit = *max_element(logits.begin(), logits.end());
    vector<float> exp_vals;
    float sum = 0;
    for (float x : logits) {
        float e = exp(x - max_logit);
        exp_vals.push_back(e);
        sum += e;
    }
    for (float& x : exp_vals) x /= sum;
    return exp_vals;
}

float cross_entropy(const vector<float>& probs, uint8_t label) {
    return -log(probs[label] + 1e-9f);
}

int argmax(const vector<float>& v) {
    return max_element(v.begin(), v.end()) - v.begin();
}

int main() {
    system("chcp 65001");
    int num_train = 500;
    auto images = read_mnist_images("train-images-idx3-ubyte", num_train);
    auto labels = read_mnist_labels("train-labels-idx1-ubyte", num_train);

    // 初始化参数
    int out_ch = 8, kernel = 3;
    ConvWeights conv_w(out_ch, vector<vector<vector<float>>>(1, vector<vector<float>>(kernel, vector<float>(kernel))));
    Biases conv_b(out_ch, 0.0f);

    for (auto& oc : conv_w)
        for (auto& ic : oc)
            for (auto& row : ic)
                for (auto& w : row)
                    w = random_weight();

    int fc_in = (28 - 2) * (28 - 2) * out_ch;
    DenseWeights fc_w(10, vector<float>(fc_in));
    Biases fc_b(10, 0.0f);

    for (auto& row : fc_w)
        for (auto& w : row)
            w = random_weight();

    float lr = 0.01;

    // 训练
    for (int epoch = 1; epoch <= 10; ++epoch) {
        float total_loss = 0;
        int correct = 0;

        for (int i = 0; i < num_train; ++i) {
            auto x = images[i].pixels;
            auto y = labels[i];

            auto conv_out = conv_forward(x, conv_w, conv_b);
            auto flat = flatten(conv_out);
            auto logits = dense_forward(flat, fc_w, fc_b);
            auto probs = softmax(logits);

            total_loss += cross_entropy(probs, y);
            if (argmax(probs) == y) correct++;

            // --- 反向传播 (SGD更新, 无梯度优化器) ---
            vector<float> dlogits(10);
            for (int j = 0; j < 10; ++j)
                dlogits[j] = probs[j] - (j == y ? 1.0f : 0.0f);

            for (int j = 0; j < 10; ++j) {
                for (int k = 0; k < fc_in; ++k)
                    fc_w[j][k] -= lr * dlogits[j] * flat[k];
                fc_b[j] -= lr * dlogits[j];
            }
        }

        cout << "[Epoch " << epoch << "] Loss: " << total_loss / num_train
             << ", Accuracy: " << (float)correct / num_train << endl;
    }

    ofstream fout("results.txt");
    for (int i = 0; i < num_train; ++i) {
        auto x = images[i].pixels;
        auto y = labels[i];

        auto conv_out = conv_forward(x, conv_w, conv_b);
        auto flat = flatten(conv_out);
        auto logits = dense_forward(flat, fc_w, fc_b);
        auto probs = softmax(logits);

        int pred = argmax(probs);
        fout << "Image " << i << ": Predicted = " << pred << ", Label = " << (int)y << endl;
    }
    fout.close();
    cout << "识别结果已保存到 results.txt！" << endl;

    ofstream fout1("results1.txt");
    for (int i = 0; i < num_train; ++i) {
        auto x = images[i].pixels;
        auto y = labels[i];

        auto conv_out = conv_forward(x, conv_w, conv_b);
        auto flat = flatten(conv_out);
        auto logits = dense_forward(flat, fc_w, fc_b);
        auto probs = softmax(logits);

        int pred = argmax(probs);

        // 保存预测结果和图像像素
        fout1 << pred << " " << (int)y;
        for (int r = 0; r < images[i].rows; ++r)
            for (int c = 0; c < images[i].cols; ++c)
                fout1 << " " << images[i].pixels[r][c];
        fout1 << endl;
    }
    fout1.close();
    return 0;
}
