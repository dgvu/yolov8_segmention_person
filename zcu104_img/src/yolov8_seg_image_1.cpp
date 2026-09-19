#include <net.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Object {
    cv::Rect2f box;
    int label;
    float score;
    std::vector<float> mask_coeff;
};

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

static std::vector<int> nms(const std::vector<Object>& objects, float iou_thres) {
    std::vector<int> indices(objects.size());
    for (int i = 0; i < (int)objects.size(); ++i) indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return objects[a].score > objects[b].score;
    });

    std::vector<int> keep;
    std::vector<bool> removed(objects.size(), false);

    for (int _i = 0; _i < (int)indices.size(); ++_i) {
        int i = indices[_i];
        if (removed[i]) continue;

        keep.push_back(i);

        for (int _j = _i + 1; _j < (int)indices.size(); ++_j) {
            int j = indices[_j];
            if (removed[j]) continue;

            if (iou(objects[i].box, objects[j].box) > iou_thres) {
                removed[j] = true;
            }
        }
    }

    return keep;
}

int main() {
    const char* param_path = "model/model.ncnn.param";
    const char* bin_path   = "model/model.ncnn.bin";
    const char* image_path = "input/test.jpg";

    const int target_size = 320;
    const float conf_thres = 0.25f;
    const float nms_thres = 0.45f;
    const int num_classes = 80;
    const int num_masks = 32;

    ncnn::Net net;

    if (net.load_param(param_path) != 0) {
        std::cerr << "Failed to load param: " << param_path << std::endl;
        return -1;
    }

    if (net.load_model(bin_path) != 0) {
        std::cerr << "Failed to load bin: " << bin_path << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << image_path << std::endl;
        return -1;
    }

    std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        img.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        img.cols,
        img.rows,
        target_size,
        target_size
    );

    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out0;
    ncnn::Mat out1;

    ex.extract("out0", out0);
    ex.extract("out1", out1);

    std::cout << "out0 shape: w=" << out0.w << ", h=" << out0.h
              << ", c=" << out0.c << ", dims=" << out0.dims << std::endl;
    std::cout << "out1 shape: w=" << out1.w << ", h=" << out1.h
              << ", c=" << out1.c << ", dims=" << out1.dims << std::endl;

    std::vector<Object> proposals;

    int num_preds = out0.w;      // 2100
    int num_attrs = out0.h;      // 116 = 4 + 80 + 32

    if (num_attrs < 4 + num_classes + num_masks) {
        std::cerr << "Unexpected output shape. num_attrs=" << num_attrs << std::endl;
        return -1;
    }

    for (int i = 0; i < num_preds; ++i) {
        float cx = out0.row(0)[i];
        float cy = out0.row(1)[i];
        float bw = out0.row(2)[i];
        float bh = out0.row(3)[i];

        int best_class = -1;
        float best_score = 0.0f;

        for (int c = 0; c < num_classes; ++c) {
            float score = out0.row(4 + c)[i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < conf_thres) continue;

        float x0 = cx - bw * 0.5f;
        float y0 = cy - bh * 0.5f;
        float x1 = cx + bw * 0.5f;
        float y1 = cy + bh * 0.5f;

        x0 = std::max(0.0f, std::min((float)target_size, x0));
        y0 = std::max(0.0f, std::min((float)target_size, y0));
        x1 = std::max(0.0f, std::min((float)target_size, x1));
        y1 = std::max(0.0f, std::min((float)target_size, y1));

        Object obj;
        obj.box = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
        obj.label = best_class;
        obj.score = best_score;

        obj.mask_coeff.resize(num_masks);
        for (int m = 0; m < num_masks; ++m) {
            obj.mask_coeff[m] = out0.row(4 + num_classes + m)[i];
        }

        proposals.push_back(obj);
    }

    std::cout << "Proposals before NMS: " << proposals.size() << std::endl;

    std::vector<int> keep = nms(proposals, nms_thres);
    std::cout << "Objects after NMS: " << keep.size() << std::endl;

    cv::Mat overlay = img.clone();

    float scale_x = (float)img.cols / target_size;
    float scale_y = (float)img.rows / target_size;

    for (int idx = 0; idx < (int)keep.size(); ++idx) {
        const Object& obj = proposals[keep[idx]];

        cv::Mat mask_small(out1.h, out1.w, CV_32FC1, cv::Scalar(0));

        for (int y = 0; y < out1.h; ++y) {
            for (int x = 0; x < out1.w; ++x) {
                float v = 0.0f;
                for (int m = 0; m < num_masks; ++m) {
                    const float* proto = out1.channel(m);
                    v += obj.mask_coeff[m] * proto[y * out1.w + x];
                }
                mask_small.at<float>(y, x) = sigmoid(v);
            }
        }

        cv::Mat mask_320;
        cv::resize(mask_small, mask_320, cv::Size(target_size, target_size));

        cv::Mat mask_orig;
        cv::resize(mask_320, mask_orig, cv::Size(img.cols, img.rows));

        cv::Mat mask_bin;
        cv::threshold(mask_orig, mask_bin, 0.5, 255, cv::THRESH_BINARY);
        mask_bin.convertTo(mask_bin, CV_8UC1);

        cv::Rect box_orig;
        box_orig.x = std::max(0, (int)(obj.box.x * scale_x));
        box_orig.y = std::max(0, (int)(obj.box.y * scale_y));
        box_orig.width = std::min(img.cols - box_orig.x, (int)(obj.box.width * scale_x));
        box_orig.height = std::min(img.rows - box_orig.y, (int)(obj.box.height * scale_y));

        cv::Mat cropped_mask = cv::Mat::zeros(mask_bin.size(), CV_8UC1);
        if (box_orig.width > 0 && box_orig.height > 0) {
            mask_bin(box_orig).copyTo(cropped_mask(box_orig));
        }

        cv::Scalar color(0, 0, 255);

        cv::Mat color_mask = cv::Mat::zeros(img.size(), CV_8UC3);
        color_mask.setTo(color, cropped_mask);

        cv::addWeighted(overlay, 1.0, color_mask, 0.45, 0, overlay);

        cv::rectangle(overlay, box_orig, color, 2);

        std::string label_text = "cls=" + std::to_string(obj.label) +
                                 " score=" + std::to_string(obj.score).substr(0, 4);
        cv::putText(overlay, label_text, cv::Point(box_orig.x, std::max(20, box_orig.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

        std::string mask_png = "output/mask_" + std::to_string(idx) + ".png";
        std::string mask_bin_path = "output/mask_" + std::to_string(idx) + ".bin";
        std::string info_path = "output/mask_" + std::to_string(idx) + "_info.txt";

        cv::imwrite(mask_png, cropped_mask);

        cv::Mat continuous_mask = cropped_mask.isContinuous() ? cropped_mask : cropped_mask.clone();
        std::ofstream fout(mask_bin_path, std::ios::binary);
        fout.write((char*)continuous_mask.data, continuous_mask.total() * continuous_mask.elemSize());
        fout.close();

        std::ofstream info(info_path);
        info << "class_id=" << obj.label << "\n";
        info << "score=" << obj.score << "\n";
        info << "box_x=" << box_orig.x << "\n";
        info << "box_y=" << box_orig.y << "\n";
        info << "box_w=" << box_orig.width << "\n";
        info << "box_h=" << box_orig.height << "\n";
        info << "mask_w=" << cropped_mask.cols << "\n";
        info << "mask_h=" << cropped_mask.rows << "\n";
        info.close();

        std::cout << "Saved " << mask_png << std::endl;
    }

    cv::imwrite("output/result_overlay.jpg", overlay);
    std::cout << "Saved output/result_overlay.jpg" << std::endl;

    return 0;
}
