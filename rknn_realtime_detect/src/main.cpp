#include <thread>
#include <vector>
#include <chrono>
// #include "debug.h"

#include <iostream>
#include <queue>
#include <string>
#include <mutex>

#include <cstring>
#include <unistd.h>

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

// extern "C"
//{
#include "postprocess.h"
// }

#include "rga.h"
#include "rknn_api.h"

using namespace std;
using namespace cv;

class MyFrame
{
public:
  cv::Mat img;
  unsigned char number;
  detect_result_group_t result;
  MyFrame(cv::Mat image, int i)
  {
    img = image;
    number = i;
  }
};

bool KeepAlive = true;

MyFrame *frame_array[60];

queue<MyFrame* > in_queue;
queue<MyFrame* > out_queue;

mutex door_in;
mutex door_out;

std::chrono::time_point<std::chrono::high_resolution_clock> first_frame_time;
std::chrono::time_point<std::chrono::high_resolution_clock> sixtith_frame_time;

int image_grabber();
int show_result();
int npu_inference(rknn_core_mask core, char *model_name);

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

inline int Waiting()
{
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  return 0;
}

inline int QueSize()
{
  std::chrono::duration<double> duration1(sixtith_frame_time - first_frame_time);
  double fps = 1 / (duration1.count() / 60);
  cout << "In_queue: " << in_queue.size() << " Out_queue: " << out_queue.size() << " FPS: " << fps << endl;
  return 0;
}

int main(int argvc, char *argv[])
{
  std::vector<std::thread> threads;
  try
  {

    threads.push_back(std::thread(image_grabber));
    threads.push_back(std::thread(show_result));

    //rknn_core_mask core_mask[] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2, RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
    rknn_core_mask core_mask[] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};

    for (auto core : core_mask)
    {
      threads.push_back(std::thread(npu_inference, core, (char *)argv[1]));
    }

    for (int i = 0; i < threads.size(); i++)
    {
      threads.at(i).join();
    }
    return 0;
  }
  catch (...)
  {
    cout << "main crash" << endl;
    KeepAlive = false;
    for (int i = 0; i < threads.size(); i++)
    {
      threads.at(i).join();
    }
    return -1;
  }
}

int image_grabber()
{
  try
  {
    VideoCapture cap("./model/in.avi");
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    int i = 0;

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
      cout << "Error opening video stream or file" << endl;
      throw;
      return -1;
    }
    cout << "opencv init" << endl;

    MyFrame *pMyFrame = nullptr;

    while (KeepAlive)
    {
      Mat orig_frame, frame;
      cap >> frame;
      cvtColor(frame, orig_frame, COLOR_BGR2RGB);
      resize(orig_frame, frame, Size(640, 640), cv::INTER_LINEAR); // will be replace on rga.resize
      if (frame.empty())
        break;

      while (in_queue.size() > 30)
      {
        Waiting();
        // cout << "full que" << endl;
      }

      pMyFrame = new MyFrame(frame, i);

      door_in.lock();
      in_queue.push(pMyFrame);
      door_in.unlock();
      // cout << "frame push into que" << endl;

      i = (i + 1) % 60; // i in (0 ... 59)
    }
    cap.release();
    return 0;
  }
  catch (...)
  {
    cout << "opencv was crashed" << endl;
    KeepAlive = false;
    return -1;
  }
}

int show_result()
{
  try
  {
    // int img_width = 640;
    // int img_height = 640; // 480

    // int width = 640;  // size AI
    // int height = 640; //

    float scale_w = 1.0; //(float)width / img_width;
    float scale_h = 1.0; //(float)height / img_height;
    int height = 640;
    int width = 640;

    // int img_channel = 0;
    const float nms_threshold = 0.45;      // NMS_THRESH;
    const float box_conf_threshold = 0.25; // BOX_THRESH;

    cv::Mat orig_img, img;
    cout << "show init" << endl;

    MyFrame *pMyFrame = nullptr;
    int count, next_el = 0;
    while (KeepAlive)
    {

      if (frame_array[next_el] != nullptr)
      {
        pMyFrame = frame_array[next_el];
        frame_array[next_el] = nullptr;
      }
      else
      {
        while (true)
        {
          while (out_queue.empty())
          {
            Waiting();
          }
          door_out.lock();
          // input_res = out_queue.front();
          pMyFrame = out_queue.front();
          out_queue.pop();
          door_out.unlock();

          if (pMyFrame->number == next_el)
          {
            break;
          }
          else
          {
            if (frame_array[pMyFrame->number] != nullptr)
            {
              throw;
              cout << "Critical array error" << endl;
            }
            frame_array[pMyFrame->number] = pMyFrame;
            pMyFrame = nullptr;
          }
        }
      }

      count = pMyFrame->number;
      if (count == 0)
      {
        first_frame_time = std::chrono::high_resolution_clock::now();
      }

      cout << count << endl;

      img = pMyFrame->img;
      cvtColor(img, orig_img, COLOR_RGB2BGR);

      detect_result_group_t detect_result_group = pMyFrame->result;

      //  Draw Objects
      char text[256];

      for (int i = 0; i < detect_result_group.count; i++)
      {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        /*
        printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom, det_result->prop);
        */
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
        putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
      }

      cv::imshow("RKNPU result viewer", orig_img);
      char c = (char)cv::waitKey(3);
      if (c == 27)
      {
        break; // (27)escape key
      }

      delete pMyFrame;
      pMyFrame = nullptr;

      if (count >= 59)
      {
        sixtith_frame_time = std::chrono::high_resolution_clock::now();
        QueSize();
      }
      next_el = (count + 1) % 60;
    }
    cv::destroyAllWindows();
    return -1;
  }
  catch (...)
  {
    cout << "show result crash" << endl;
    KeepAlive = false;
    return -1;
  }
}
int npu_inference(rknn_core_mask core, char *model_name)
{
  try
  {
    int ret;
    rknn_context ctx;
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);
    cout << "load model" << endl;

    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    // cout << "rknn init" << endl;
    if (ret < 0)
    {
      printf("rknn_init error ret=%d\n", ret);
      throw;
      return -1;
    }

    rknn_core_mask core_qwe = core;
    ret = rknn_set_core_mask(ctx, core_qwe);
    if (ret < 0)
    {
      printf("rknn_core_mask error ret=%d\n", ret);
      throw;
      return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
      printf("rknn_init error ret=%d\n", ret);
      throw;
      return -1;
    }
    // printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(rknn_input_output_num));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
      printf("rknn_init error ret=%d\n", ret);
      throw;
      return -1;
    }
    // printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
      input_attrs[i].index = i;
      ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
      if (ret < 0)
      {
        // printf("rknn_init error ret=%d\n", ret);
        throw;
        return -1;
      }
      // printf(); show input attr
      // dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
      output_attrs[i].index = i;
      ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
      // dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
      printf("model is NCHW input fmt\n");
      channel = input_attrs[0].dims[1];
      height = input_attrs[0].dims[2];
      width = input_attrs[0].dims[3];
    }
    else
    {
      printf("model is NHWC input fmt\n");
      height = input_attrs[0].dims[1];
      width = input_attrs[0].dims[2];
      channel = input_attrs[0].dims[3];
    }

    // printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // inputs[0].buf = (void*)img.data;
    cout << "rknn full init" << endl;

    float scale_w = 1.0, scale_h = 1.0;
    const float box_conf_threshold = 0.25;
    const float nms_threshold = 0.45;

    MyFrame *pMyFrame = nullptr;
    while (KeepAlive)
    {

      door_in.lock();
      while (in_queue.empty())
      {
        Waiting();
        // cout << "empty que in npu" << endl;
      }
      pMyFrame = in_queue.front();
      in_queue.pop();
      door_in.unlock();

      //cv::Mat img = pMyFrame->img;
      // inputs[0].buf = (void *)img.data;
      
      inputs[0].buf = (void *)(pMyFrame->img.data);
      rknn_inputs_set(ctx, io_num.n_input, inputs);

      rknn_output outputs[io_num.n_output];
      memset(outputs, 0, sizeof(outputs));
      for (int i = 0; i < io_num.n_output; i++)
      {
        outputs[i].want_float = 0;
      }

      ret = rknn_run(ctx, NULL);
      ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

      std::vector<float> out_scales;
      std::vector<int32_t> out_zps;
      // std::vector<int8_t> out_buf;
      for (int i = 0; i < io_num.n_output; ++i)
      {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
        // out_buf.push_back(*(int8_t*)outputs[i].buf);
      }

      detect_result_group_t detect_result_group;
      // printf("before post_process\n");
      post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                   box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

      pMyFrame->result = detect_result_group;

      door_out.lock();
      out_queue.push(pMyFrame);
      door_out.unlock();

      ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    }
    // rknn.destroy

    free(model_data);
    return -1;
  }
  catch (...)
  {

    cout << "npu crash" << endl;
    KeepAlive = false;
    return -1;
  }
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)
  {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
  FILE *fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++)
  {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}
