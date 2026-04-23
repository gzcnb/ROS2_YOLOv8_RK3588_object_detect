#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

// 默认打开 USB 摄像头 (/dev/video0)
#define VIDEO_PATH 0 

#define OBJ_THRESH 0.25
#define NMS_THRESH 0.45
#define REG_MAX    16
#define OBJ_CLS_HAS_SIGMOID 1 

const char* CLASS_NAMES[] = { "lemon" };

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} DetectResult;

// ==================== 后处理算法区 ====================
static void dfl_decode(const float* box_pred, float* box_out) {
    for (int i = 0; i < 4; i++) {
        const float* ptr = box_pred + i * REG_MAX;
        float max_val = ptr[0];
        for (int j = 1; j < REG_MAX; j++) { if (ptr[j] > max_val) max_val = ptr[j]; }
        float exp_sum = 0, softmax[REG_MAX];
        for (int j = 0; j < REG_MAX; j++) { softmax[j] = expf(ptr[j] - max_val); exp_sum += softmax[j]; }
        for (int j = 0; j < REG_MAX; j++) softmax[j] /= exp_sum;
        float val = 0;
        for (int j = 0; j < REG_MAX; j++) val += j * softmax[j];
        box_out[i] = val;
    }
}

static void process_scale(const float* dfl_data, const float* obj_data, const float* cls_data,
                          int H, int W, int stride, std::vector<DetectResult>& detections, float obj_thresh) {
    int grid_size = H * W;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int idx = h * W + w;
            float obj_val = obj_data[idx];
            float cls_val = cls_data[idx];
#if OBJ_CLS_HAS_SIGMOID
#else
            obj_val = 1.0f / (1.0f + expf(-obj_val));
            cls_val = 1.0f / (1.0f + expf(-cls_val));
#endif
            float score = obj_val * cls_val;
            if (score < obj_thresh) continue;

            float box_pred[64];
            for (int c = 0; c < 64; c++) box_pred[c] = dfl_data[c * grid_size + idx];
            float box[4]; dfl_decode(box_pred, box);
            float cx = (w + 0.5f) * stride;
            float cy = (h + 0.5f) * stride;

            detections.push_back({ cx - box[0] * stride, cy - box[1] * stride,
                                   cx + box[2] * stride, cy + box[3] * stride, score, 0 });
        }
    }
}

float calculate_iou(float* b1, float* b2) {
    float x1 = std::max(b1[0], b2[0]), y1 = std::max(b1[1], b2[1]);
    float x2 = std::min(b1[2], b2[2]), y2 = std::min(b1[3], b2[3]);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    float a2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    return inter / (a1 + a2 - inter + 1e-6f);
}

void nms(std::vector<DetectResult>& dets, float thresh) {
    std::sort(dets.begin(), dets.end(), [](const DetectResult& a, const DetectResult& b) { return a.score > b.score; });
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<DetectResult> result;
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            float b1[4] = {dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2};
            float b2[4] = {dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2};
            if (calculate_iou(b1, b2) > thresh) suppressed[j] = true;
        }
    }
    dets = result;
}
// =========================================================

int main() {
    int ret; rknn_context ctx = 0;

    // 1. 加载模型
    const char* model_path = "/home/orangepi/ros2_ws/src/my_text_package/include/librknn_api/yolov8n.rknn";   //检测模型路径
    FILE* fp = fopen(model_path, "rb");
    if (!fp) { printf("Failed to open model\n"); return -1; }
    fseek(fp, 0, SEEK_END); uint32_t model_size = ftell(fp); fseek(fp, 0, SEEK_SET);
    unsigned char* model_data = (unsigned char*)malloc(model_size);
    fread(model_data, 1, model_size, fp); fclose(fp);
    ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) return -1;

    // 2. 获取模型信息
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    int num_outputs = io_num.n_output;

    rknn_tensor_attr input_attr; input_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    int model_input_w = input_attr.dims[2];
    int model_input_h = input_attr.dims[1];

    rknn_tensor_attr output_attrs[9]; int out_H[9], out_W[9];
    for (int i = 0; i < num_outputs; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        out_H[i] = (output_attrs[i].n_dims == 4) ? output_attrs[i].dims[2] : output_attrs[i].dims[3];
        out_W[i] = (output_attrs[i].n_dims == 4) ? output_attrs[i].dims[3] : output_attrs[i].dims[4];
    }

    // 3. 准备图像内存和 RKNN 结构体
    cv::Mat padded_img(model_input_h, model_input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat rgb_img(model_input_h, model_input_w, CV_8UC3);

    rknn_input inputs[1]; 
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0; 
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    rknn_output outputs[9]; 
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < num_outputs; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0; // 让驱动自动管理内存，最稳妥
    }

    // 4. 打开视频流
    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        printf("Failed to open video stream!\n");
        return -1;
    }

    printf("Start detecting... Press 'q' to quit.\n");

    cv::Mat orig_img;
    while (cap.isOpened()) {
        cap >> orig_img;
        if (orig_img.empty()) break;

        int orig_w = orig_img.cols, orig_h = orig_img.rows;
        float scale = std::min((float)model_input_w / orig_w, (float)model_input_h / orig_h);
        int new_w = (int)(orig_w * scale), new_h = (int)(orig_h * scale);
        int pad_w = model_input_w - new_w, pad_h = model_input_h - new_h;

        // 高效 Letterbox
        cv::Mat roi = padded_img(cv::Rect(pad_w/2, pad_h-pad_h/2, new_w, new_h));
        cv::resize(orig_img, roi, cv::Size(new_w, new_h));
        cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);

        // 每帧绑定最新数据并喂给 NPU
        inputs[0].buf = rgb_img.data; 
        inputs[0].size = rgb_img.cols * rgb_img.rows * rgb_img.channels();
        rknn_inputs_set(ctx, 1, inputs);

        // 推理 & 获取输出
        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, num_outputs, outputs, NULL);

        // 后处理
        int strides[3] = {8, 16, 32};
        std::vector<DetectResult> detections;
        for (int s = 0; s < 3; s++) {
            process_scale((float*)outputs[s*3].buf, (float*)outputs[s*3+1].buf, (float*)outputs[s*3+2].buf,
                          out_H[s*3], out_W[s*3], strides[s], detections, OBJ_THRESH);
        }
        nms(detections, NMS_THRESH);

        // 画框
        float pad_x = (model_input_w - orig_w * scale) / 2.0f;
        float pad_y = (model_input_h - orig_h * scale) / 2.0f;
        for (size_t i = 0; i < detections.size(); i++) {
            float x1 = std::max(0.0f, std::min((detections[i].x1 - pad_x) / scale, (float)orig_w));
            float y1 = std::max(0.0f, std::min((detections[i].y1 - pad_y) / scale, (float)orig_h));
            float x2 = std::max(0.0f, std::min((detections[i].x2 - pad_x) / scale, (float)orig_w));
            float y2 = std::max(0.0f, std::min((detections[i].y2 - pad_y) / scale, (float)orig_h));

            cv::rectangle(orig_img, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(0,255,0), 2);
            char label[128]; 
            snprintf(label, sizeof(label), "lemon %.2f", detections[i].score);
            cv::putText(orig_img, label, cv::Point(x1, y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
        }

        cv::imshow("YOLOv8 Lemon Detection", orig_img);
        rknn_outputs_release(ctx, num_outputs, outputs);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // 5. 清理资源
    cap.release();
    cv::destroyAllWindows();
    rknn_destroy(ctx);
    return 0;
}
