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

static float rect_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
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

            if (rect_iou(objects[i].box, objects[j].box) > iou_thres) {
                removed[j] = true;
            }
        }
    }

    return keep;
}

static bool process_frame(
    ncnn::Net& net,
    const cv::Mat& img,
    cv::Mat& overlay,
    int frame_id,
    bool save_masks
) {
    const int target_size = 320;
    const float conf_thres = 0.25f;
    const float nms_thres = 0.45f;
    const int num_classes = 80;
    const int num_masks = 32;

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

    if (ex.extract("out0", out0) != 0) {
        std::cerr << "Failed to extract out0\n";
        return false;
    }

    if (ex.extract("out1", out1) != 0) {
        std::cerr << "Failed to extract out1\n";
        return false;
    }

    int num_preds = out0.w;
    int num_attrs = out0.h;

    if (num_attrs < 4 + num_classes + num_masks) {
        std::cerr << "Unexpected out0 shape: w=" << out0.w
                  << ", h=" << out0.h << ", c=" << out0.c << "\n";
        return false;
    }

    std::vector<Object> proposals;

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

        if (x1 <= x0 || y1 <= y0) continue;

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

    std::vector<int> keep = nms(proposals, nms_thres);

    overlay = img.clone();

    float scale_x = (float)img.cols / target_size;
    float scale_y = (float)img.rows / target_size;

    int saved_mask_count = 0;

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

        if (box_orig.width <= 0 || box_orig.height <= 0) continue;

        cv::Mat cropped_mask = cv::Mat::zeros(mask_bin.size(), CV_8UC1);
        mask_bin(box_orig).copyTo(cropped_mask(box_orig));

        cv::Scalar color(0, 0, 255);

        cv::Mat color_mask = cv::Mat::zeros(img.size(), CV_8UC3);
        color_mask.setTo(color, cropped_mask);

        cv::addWeighted(overlay, 1.0, color_mask, 0.45, 0, overlay);
        cv::rectangle(overlay, box_orig, color, 2);

        std::string label_text = "cls=" + std::to_string(obj.label) +
                                 " score=" + std::to_string(obj.score).substr(0, 4);

        cv::putText(
            overlay,
            label_text,
            cv::Point(box_orig.x, std::max(20, box_orig.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        );

        if (save_masks && saved_mask_count < 1) {
            char mask_png[256];
            char mask_bin_path[256];
            char info_path[256];

            snprintf(mask_png, sizeof(mask_png), "output/video_mask_f%05d_obj%d.png", frame_id, idx);
            snprintf(mask_bin_path, sizeof(mask_bin_path), "output/video_mask_f%05d_obj%d.bin", frame_id, idx);
            snprintf(info_path, sizeof(info_path), "output/video_mask_f%05d_obj%d_info.txt", frame_id, idx);

            cv::imwrite(mask_png, cropped_mask);

            cv::Mat continuous_mask = cropped_mask.isContinuous() ? cropped_mask : cropped_mask.clone();
            std::ofstream fout(mask_bin_path, std::ios::binary);
            fout.write((char*)continuous_mask.data, continuous_mask.total() * continuous_mask.elemSize());
            fout.close();

            std::ofstream info(info_path);
            info << "frame_id=" << frame_id << "\n";
            info << "class_id=" << obj.label << "\n";
            info << "score=" << obj.score << "\n";
            info << "box_x=" << box_orig.x << "\n";
            info << "box_y=" << box_orig.y << "\n";
            info << "box_w=" << box_orig.width << "\n";
            info << "box_h=" << box_orig.height << "\n";
            info << "mask_w=" << cropped_mask.cols << "\n";
            info << "mask_h=" << cropped_mask.rows << "\n";
            info.close();

            saved_mask_count++;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    std::string video_path = "input/test_video.mp4";

    if (argc >= 2) {
        video_path = argv[1];
    }

    const char* param_path = "model/model.ncnn.param";
    const char* bin_path   = "model/model.ncnn.bin";

    ncnn::Net net;

    if (net.load_param(param_path) != 0) {
        std::cerr << "Failed to load param: " << param_path << std::endl;
        return -1;
    }

    if (net.load_model(bin_path) != 0) {
        std::cerr << "Failed to load bin: " << bin_path << std::endl;
        return -1;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return -1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    if (fps <= 1.0 || fps > 120.0) fps = 10.0;

    std::cout << "Video opened: " << video_path << "\n";
    std::cout << "Size: " << width << "x" << height << ", fps=" << fps << "\n";

    cv::VideoWriter writer;
    writer.open(
        "output/result_video.avi",
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps,
        cv::Size(width, height)
    );

    bool writer_ok = writer.isOpened();
    if (!writer_ok) {
        std::cerr << "Warning: Cannot open VideoWriter. Will save frames as JPG instead.\n";
    }

    int frame_id = 0;
    int max_frames = -1;        // đổi số này nếu muốn chạy nhiều hơn
    int save_mask_every = 10;    // lưu mask mỗi 10 frame để tránh đầy SD

if (argc >= 3) {
    max_frames = std::stoi(argv[2]);
}

    cv::TickMeter total_timer;
    total_timer.start();

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        cv::Mat overlay;

        bool save_masks = (frame_id % save_mask_every == 0);

        cv::TickMeter tm;
        tm.start();

        bool ok = process_frame(net, frame, overlay, frame_id, save_masks);

        tm.stop();

        if (!ok) {
            std::cerr << "Failed at frame " << frame_id << "\n";
            break;
        }

        double ms = tm.getTimeMilli();

        std::string fps_text = "frame=" + std::to_string(frame_id) +
                               " time=" + std::to_string(ms).substr(0, 5) + " ms";

        cv::putText(
            overlay,
            fps_text,
            cv::Point(20, 35),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(255, 255, 255),
            2
        );

        if (writer_ok) {
            writer.write(overlay);
        } else {
            char frame_path[256];
            snprintf(frame_path, sizeof(frame_path), "output/video_frame_%05d.jpg", frame_id);
            cv::imwrite(frame_path, overlay);
        }

        std::cout << "Processed frame " << frame_id << ", " << ms << " ms\n";

        frame_id++;

        if (max_frames > 0 && frame_id >= max_frames) {
            std::cout << "Reached max_frames=" << max_frames << "\n";
            break;
        }
    }

    total_timer.stop();

    cap.release();
    writer.release();

    double total_sec = total_timer.getTimeSec();
    double avg_fps = frame_id / std::max(0.001, total_sec);

    std::cout << "Done.\n";
    std::cout << "Processed frames: " << frame_id << "\n";
    std::cout << "Total time: " << total_sec << " sec\n";
    std::cout << "Average FPS: " << avg_fps << "\n";
    std::cout << "Saved: output/result_video.avi\n";

    return 0;
}
