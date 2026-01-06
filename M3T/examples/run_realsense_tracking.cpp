// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)
// Modified for RealSense camera tracking

#include <filesystem/filesystem.h>
#include <m3t/basic_depth_renderer.h>
#include <m3t/body.h>
#include <m3t/common.h>
#include <m3t/constraint.h>
#include <m3t/depth_model.h>
#include <m3t/depth_modality.h>
#include <m3t/link.h>
#include <m3t/normal_viewer.h>
#include <m3t/optimizer.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/realsense_camera.h>
#include <m3t/renderer_geometry.h>
#include <m3t/silhouette_renderer.h>
#include <m3t/static_detector.h>
#include <m3t/texture_modality.h>
#include <m3t/tracker.h>

#include <Eigen/Geometry>
#include <fstream>
#include <memory>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
  // 魔方参数
  const std::string body_name = "cube";
  const std::filesystem::path obj_file_path = "/media/jyj/JYJ/RBOT-dataset/cube/cube_scaled.obj";
  const float cube_size_m = 0.056f;  // 56mm = 0.056m
  const float initial_distance = 0.45f;  // 45cm = 0.45m
  
  // 输出目录设置
  const std::filesystem::path output_directory = "/tmp/m3t_tracking_output";
  std::filesystem::create_directories(output_directory);
  
  // 检查 OBJ 文件是否存在
  if (!std::filesystem::exists(obj_file_path)) {
    std::cerr << "Error: OBJ file not found at " << obj_file_path << std::endl;
    return -1;
  }
  
  // 创建临时目录用于存储模型文件
  const std::filesystem::path temp_directory = "/tmp/m3t_cube_models";
  std::filesystem::create_directories(temp_directory);
  
  // 设置 tracker 和 renderer geometry
  auto tracker_ptr{std::make_shared<m3t::Tracker>("tracker")};
  auto renderer_geometry_ptr{
      std::make_shared<m3t::RendererGeometry>("renderer_geometry")};
  
  // 设置 RealSense 相机（颜色、深度和IR相机）
  // 深度相机使用颜色相机作为世界坐标系，使两个视角对齐
  // 启用IR相机用于记录位姿（不用于追踪）
  m3t::RealSense::GetInstance().UseIRCamera(1);  // 使用左IR相机（索引1）
  
  auto color_camera_ptr{
      std::make_shared<m3t::RealSenseColorCamera>("realsense_color", false)};
  auto depth_camera_ptr{
      std::make_shared<m3t::RealSenseDepthCamera>("realsense_depth", true)};  // true = use_color_as_world_frame
  // IR相机使用颜色相机作为世界坐标系，用于坐标变换
  auto ir_camera_ptr{
      std::make_shared<m3t::RealSenseIRCamera>("realsense_ir", 1, true)};  // ir_index=1, use_color_as_world_frame=true
  
  // 关闭Emitter以避免IR图像中出现结构光光点
  // 注意：这可能会略微影响深度相机的精度，但IR图像会更清晰
  m3t::RealSense::GetInstance().SetEmitterEnabled(false);
  
  // 设置 viewer（颜色和深度视图）
  auto color_viewer_ptr{std::make_shared<m3t::NormalColorViewer>(
      "color_viewer", color_camera_ptr, renderer_geometry_ptr)};
  tracker_ptr->AddViewer(color_viewer_ptr);
  auto depth_viewer_ptr{std::make_shared<m3t::NormalDepthViewer>(
      "depth_viewer", depth_camera_ptr, renderer_geometry_ptr, 0.1f, 1.0f)};
  tracker_ptr->AddViewer(depth_viewer_ptr);
  // IR viewer将在IR相机SetUp之后创建
  std::shared_ptr<m3t::NormalColorViewer> ir_viewer_ptr;
  
  // 创建 Body 对象
  float geometry_unit_in_meter = 1.0f;
  auto body_ptr{std::make_shared<m3t::Body>(
      body_name, obj_file_path, geometry_unit_in_meter, true, true,
      m3t::Transform3fA::Identity())};
  renderer_geometry_ptr->AddBody(body_ptr);
  
  // 输出 Body 信息用于调试
  if (!body_ptr->SetUp()) {
    std::cerr << "Failed to set up body" << std::endl;
    return -1;
  }
  std::cout << "Body maximum diameter: " << body_ptr->maximum_body_diameter() << " meters" << std::endl;
  
  // 设置初始姿态：在相机前 20cm 处，朝向相机
  m3t::Transform3fA initial_pose = m3t::Transform3fA::Identity();
  initial_pose.translation() = Eigen::Vector3f(0.0f, 0.0f, initial_distance);
  body_ptr->set_body2world_pose(initial_pose);
  
  // 创建模型文件路径（区域模型和深度模型）
  std::filesystem::path region_model_path = temp_directory / (body_name + "_region_model.bin");
  std::filesystem::path depth_model_path = temp_directory / (body_name + "_depth_model.bin");
  
  // 创建 region model
  std::cout << "Creating region model..." << std::endl;
  auto region_model_ptr{std::make_shared<m3t::RegionModel>(
      body_name + "_region_model", body_ptr, region_model_path,
      0.8f, 4, 500, 0.05f, 0.002f, false, 2000)};
  std::cout << "Region model created." << std::endl;
  
  // 创建 depth model
  std::cout << "Creating depth model..." << std::endl;
  auto depth_model_ptr{std::make_shared<m3t::DepthModel>(
      body_name + "_depth_model", body_ptr, depth_model_path,
      0.8f, 4, 500, 0.05f, 0.002f, false, 2000)};
  std::cout << "Depth model created." << std::endl;
  
  // 创建 region modality - 使用默认参数
  std::cout << "Creating region modality..." << std::endl;
  auto region_modality_ptr{std::make_shared<m3t::RegionModality>(
      body_name + "_region_modality", body_ptr, color_camera_ptr,
      region_model_ptr)};
  // 启用深度遮挡测量
  region_modality_ptr->MeasureOcclusions(depth_camera_ptr);
  std::cout << "Region modality created." << std::endl;
  
  // 创建深度渲染器（用于遮挡建模）
  auto color_depth_renderer_ptr{
      std::make_shared<m3t::FocusedBasicDepthRenderer>(
          "color_depth_renderer", renderer_geometry_ptr, color_camera_ptr)};
  color_depth_renderer_ptr->AddReferencedBody(body_ptr);
  auto depth_depth_renderer_ptr{
      std::make_shared<m3t::FocusedBasicDepthRenderer>(
          "depth_depth_renderer", renderer_geometry_ptr, depth_camera_ptr)};
  depth_depth_renderer_ptr->AddReferencedBody(body_ptr);
  
  // 创建 silhouette renderer（用于特征点模块）
  auto color_silhouette_renderer_ptr{
      std::make_shared<m3t::FocusedSilhouetteRenderer>(
          "color_silhouette_renderer", renderer_geometry_ptr,
          color_camera_ptr)};
  color_silhouette_renderer_ptr->AddReferencedBody(body_ptr);
  
  // 创建 texture modality（特征点模块）- 使用默认参数
  std::cout << "Creating texture modality..." << std::endl;
  auto texture_modality_ptr{std::make_shared<m3t::TextureModality>(
      body_name + "_texture_modality", body_ptr, color_camera_ptr,
      color_silhouette_renderer_ptr)};
  // 使用默认参数，只设置描述符类型为ORB（默认值）
  texture_modality_ptr->set_descriptor_type(m3t::TextureModality::DescriptorType::ORB);
  std::cout << "Texture modality created." << std::endl;
  
  // 创建 depth modality（深度模块）- 使用默认参数
  std::cout << "Creating depth modality..." << std::endl;
  auto depth_modality_ptr{std::make_shared<m3t::DepthModality>(
      body_name + "_depth_modality", body_ptr, depth_camera_ptr,
      depth_model_ptr)};
  // 启用深度遮挡测量
  depth_modality_ptr->MeasureOcclusions();
  // 启用深度遮挡建模
  depth_modality_ptr->ModelOcclusions(depth_depth_renderer_ptr);
  std::cout << "Depth modality created." << std::endl;
  
  // 创建 link（添加区域模态、深度模态和特征点模态）
  std::cout << "Creating link..." << std::endl;
  auto link_ptr{std::make_shared<m3t::Link>(body_name + "_link", body_ptr)};
  link_ptr->AddModality(region_modality_ptr);
  link_ptr->AddModality(depth_modality_ptr);
  link_ptr->AddModality(texture_modality_ptr);  // 启用特征点模态
  std::cout << "Link created." << std::endl;
  
  // 创建 optimizer - 使用默认参数
  std::cout << "Creating optimizer..." << std::endl;
  auto optimizer_ptr{
      std::make_shared<m3t::Optimizer>(body_name + "_optimizer", link_ptr)};
  // 使用默认参数：tikhonov_parameter_rotation = 1000.0f, tikhonov_parameter_translation = 30000.0f
  tracker_ptr->AddOptimizer(optimizer_ptr);
  std::cout << "Optimizer created." << std::endl;
  
  // 创建 detector（使用初始姿态）
  std::cout << "Creating detector..." << std::endl;
  auto detector_ptr{std::make_shared<m3t::StaticDetector>(
      body_name + "_detector", optimizer_ptr, initial_pose)};
  tracker_ptr->AddDetector(detector_ptr);
  std::cout << "Detector created." << std::endl;
  
  // 使用默认参数：n_corr_iterations = 5, n_update_iterations = 2
  
  std::cout << "Setting up tracker (this may take a while for model generation)..." << std::endl;
  if (!tracker_ptr->SetUp()) {
    std::cerr << "Failed to set up tracker" << std::endl;
    return -1;
  }
  std::cout << "Tracker setup complete!" << std::endl;
  
  // 关闭Emitter以避免IR图像中出现结构光光点
  // 注意：需要在tracker SetUp之后操作，因为此时RealSense已经初始化
  try {
    rs2::device dev = m3t::RealSense::GetInstance().profile().get_device();
    auto depth_sensor = dev.first<rs2::depth_sensor>();
    if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
      depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.0f);
      std::cout << "RealSense emitter (structured light) disabled for IR images" << std::endl;
    }
  } catch (const rs2::error &e) {
    std::cerr << "Warning: Could not disable emitter option: " << e.what() << std::endl;
  }
  
  // SetUp IR相机（必须在tracker SetUp之后，因为需要RealSense实例已经初始化）
  std::cout << "Setting up IR camera..." << std::endl;
  if (!ir_camera_ptr->SetUp()) {
    std::cerr << "Failed to set up IR camera" << std::endl;
    return -1;
  }
  std::cout << "IR camera setup complete!" << std::endl;
  
  // 创建IR viewer（用于在IR图像上渲染预测位姿）
  // 必须在IR相机SetUp之后创建，因为viewer需要访问相机的内参等信息
  ir_viewer_ptr = std::make_shared<m3t::NormalColorViewer>(
      "ir_viewer", ir_camera_ptr, renderer_geometry_ptr);
  if (!ir_viewer_ptr->SetUp()) {
    std::cerr << "Failed to set up IR viewer" << std::endl;
    return -1;
  }
  tracker_ptr->AddViewer(ir_viewer_ptr);
  std::cout << "IR viewer setup complete!" << std::endl;
  
  std::cout << "========================================" << std::endl;
  std::cout << "RealSense Tracking Setup Complete!" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Using DEFAULT parameters:" << std::endl;
  std::cout << "  Tracker:" << std::endl;
  std::cout << "    - Correspondence iterations: " << tracker_ptr->n_corr_iterations() << " (default: 5)" << std::endl;
  std::cout << "    - Update iterations: " << tracker_ptr->n_update_iterations() << " (default: 2)" << std::endl;
  std::cout << "  Optimizer:" << std::endl;
  std::cout << "    - Tikhonov translation: " << optimizer_ptr->tikhonov_parameter_translation() << " (default: 30000.0)" << std::endl;
  std::cout << "    - Tikhonov rotation: " << optimizer_ptr->tikhonov_parameter_rotation() << " (default: 1000.0)" << std::endl;
  std::cout << "  Region Modality:" << std::endl;
  std::cout << "    - Learning rate: " << region_modality_ptr->learning_rate() << " (default: 1.3)" << std::endl;
  std::cout << "    - Scales: [6, 4, 2, 1] (default)" << std::endl;
  std::cout << "    - Standard deviations: [15.0, 5.0, 3.5, 1.5] (default)" << std::endl;
  std::cout << "  Texture Modality:" << std::endl;
  std::cout << "    - Descriptor type: ORB (default)" << std::endl;
  std::cout << "    - Standard deviations: [15.0, 5.0] (default)" << std::endl;
  std::cout << "  Depth Modality:" << std::endl;
  std::cout << "    - Standard deviations: [0.05, 0.03, 0.02] (default)" << std::endl;
  std::cout << "\nInitial pose set at " << initial_distance << "m in front of camera" << std::endl;
  std::cout << "A reference model is being rendered." << std::endl;
  std::cout << "Please align your cube with the rendered model." << std::endl;
  std::cout << "Press SPACE to start detection and tracking." << std::endl;
  std::cout << "Press ESC or 'q' to exit." << std::endl;
  std::cout << "\nUsing modalities:" << std::endl;
  std::cout << "  - Region Modality: Enabled" << std::endl;
  std::cout << "  - Depth Modality: Enabled" << std::endl;
  std::cout << "  - Texture Modality: Enabled" << std::endl;
  std::cout << "\nOutput will be saved to: " << output_directory << std::endl;
  std::cout << "  - Images: combined_image_*.png (RGB + Depth side by side)" << std::endl;
  std::cout << "  - Video: tracking_video.mp4 (generated automatically)" << std::endl;
  std::cout << "  - RGB Poses: pose.txt (RGB camera coordinate system)" << std::endl;
  std::cout << "  - IR Poses: pose_ir.txt (IR camera coordinate system)" << std::endl;
  std::cout << "  - IR Images: realsense_ir_image_*.png (raw IR images)" << std::endl;
  std::cout << "  - IR Viewer Images: ir_viewer_image_*.png (IR images with rendered pose)" << std::endl;
  std::cout << "\nNote: IR camera is enabled for pose recording and image saving only, not used for tracking." << std::endl;
  std::cout << "========================================" << std::endl;
  
  // 第一阶段：渲染参考模型，等待用户对齐
  bool start_tracking = false;
  int frame_count = 0;
  
  while (!start_tracking) {
    // 更新相机图像（颜色和深度相机）
    // 注意：IR相机在tracker SetUp之后才SetUp，所以这里不更新
    if (!color_camera_ptr->UpdateImage(true)) continue;
    depth_camera_ptr->UpdateImage(true);  // 更新深度相机图像
    
    // 更新 viewer（这会渲染参考模型）
    color_viewer_ptr->UpdateViewer(frame_count);
    depth_viewer_ptr->UpdateViewer(frame_count);  // 更新深度viewer
    
    // 检查按键
    int key = cv::waitKey(1) & 0xFF;
    if (key == 32) {  // SPACE
      start_tracking = true;
      std::cout << "Starting detection and tracking..." << std::endl;
    } else if (key == 27) {  // ESC
      std::cout << "Exiting..." << std::endl;
      return 0;
    }
    
    frame_count++;
  }
  
  // 执行检测 - StaticDetector 使用 optimizer 的名称，不是 body 的名称
  // 检测前，更新detector的位姿为当前body的位姿（用户可能在对齐阶段移动了魔方）
  detector_ptr->set_link2world_pose(body_ptr->body2world_pose());
  
  std::set<std::string> detect_names{optimizer_ptr->name()};
  std::set<std::string> detected_names;
  if (!tracker_ptr->DetectPoses(detect_names, &detected_names)) {
    std::cerr << "Detection failed" << std::endl;
    return -1;
  }
  if (detected_names.empty()) {
    std::cerr << "No bodies detected. Expected optimizer name: " << optimizer_ptr->name() << std::endl;
    return -1;
  }
  std::cout << "Detection successful. Detected optimizer: " << *detected_names.begin() << std::endl;
  std::cout << "Initial pose: translation=[" << body_ptr->body2world_pose().translation().transpose() << "]" << std::endl;
  
  // 启用viewer的图像保存，以便获取包含渲染模型的图像
  color_viewer_ptr->StartSavingImages(output_directory, "png");
  depth_viewer_ptr->StartSavingImages(output_directory, "png");
  ir_viewer_ptr->StartSavingImages(output_directory, "png");  // IR viewer保存渲染后的IR图像
  
  // 启用IR相机图像保存（注意：StartSavingImages会将set_up_设为false，需要重新SetUp）
  ir_camera_ptr->StartSavingImages(output_directory, 0, "png");
  // 重新SetUp IR相机（因为StartSavingImages会重置set_up_标志）
  if (!ir_camera_ptr->SetUp()) {
    std::cerr << "Failed to re-setup IR camera after StartSavingImages" << std::endl;
    return -1;
  }
  
  // 打开位姿文件（保存为pose.txt和pose_ir.txt）
  std::filesystem::path pose_file_path = output_directory / "pose.txt";
  std::filesystem::path pose_ir_file_path = output_directory / "pose_ir.txt";
  std::ofstream pose_ofs(pose_file_path);
  std::ofstream pose_ir_ofs(pose_ir_file_path);
  if (!pose_ofs.is_open()) {
    std::cerr << "Failed to open pose file: " << pose_file_path << std::endl;
    return -1;
  }
  if (!pose_ir_ofs.is_open()) {
    std::cerr << "Failed to open IR pose file: " << pose_ir_file_path << std::endl;
    return -1;
  }
  
  // 写入位姿文件头
  pose_ofs << "# Frame Index, Timestamp (ms), Translation (x, y, z), Rotation Matrix (3x3)\n";
  pose_ofs << "# Format: frame_index timestamp tx ty tz r00 r01 r02 r10 r11 r12 r20 r21 r22\n";
  pose_ofs << "# Coordinate system: RGB camera (world frame)\n";
  
  pose_ir_ofs << "# Frame Index, Timestamp (ms), Translation (x, y, z), Rotation Matrix (3x3)\n";
  pose_ir_ofs << "# Format: frame_index timestamp tx ty tz r00 r01 r02 r10 r11 r12 r20 r21 r22\n";
  pose_ir_ofs << "# Coordinate system: IR camera\n";
  
  // 计算从RGB坐标系到IR坐标系的变换
  // IR相机的camera2world_pose是ir2color_pose（当use_color_as_world_frame=true时）
  // 从RGB到IR的变换 = (ir2color_pose)^(-1) = color2ir_pose
  m3t::Transform3fA color2ir_pose = ir_camera_ptr->camera2world_pose().inverse();
  std::cout << "Color to IR pose transformation calculated." << std::endl;
  
  // 启动模态
  for (const auto &renderer_ptr : tracker_ptr->start_modality_renderer_ptrs())
    renderer_ptr->StartRendering();
  for (const auto &modality_ptr : link_ptr->modality_ptrs())
    modality_ptr->StartModality(0, 0);
  
  // 第二阶段：自定义tracking循环，参考RBOT的方式
  int iteration = 0;
  bool tracking = true;
  
  std::cout << "Tracking started. Press ESC or 'q' to stop." << std::endl;
  std::cout << "Press 'r' or 'R' to re-detect and re-align the model." << std::endl;
  
  // 保存上一次的位姿，用于检测位姿变化
  m3t::Transform3fA previous_pose = body_ptr->body2world_pose();
  int frames_since_last_detection = 0;
  
  while (tracking) {
    // 更新相机（包括IR相机）
    tracker_ptr->UpdateCameras(iteration);
    // 更新IR相机图像（仅用于记录位姿和保存图像）
    if (!ir_camera_ptr->UpdateImage(true)) {
      // 如果UpdateImage失败，打印警告但继续追踪
      if (iteration % 100 == 0) {
        std::cerr << "Warning: Failed to update IR camera image at frame " << iteration << std::endl;
      }
    }
    
    // 计算对应点
    for (int corr_iteration = 0; corr_iteration < tracker_ptr->n_corr_iterations(); ++corr_iteration) {
      tracker_ptr->CalculateCorrespondences(iteration, corr_iteration);
      tracker_ptr->VisualizeCorrespondences(
          iteration * tracker_ptr->n_corr_iterations() + corr_iteration);
      
      // 计算梯度和Hessian，然后优化
      for (int update_iteration = 0; update_iteration < tracker_ptr->n_update_iterations(); ++update_iteration) {
        tracker_ptr->CalculateGradientAndHessian(iteration, corr_iteration, update_iteration);
        tracker_ptr->CalculateOptimization(iteration, corr_iteration, update_iteration);
        tracker_ptr->VisualizeOptimization(
            (iteration * tracker_ptr->n_corr_iterations() + corr_iteration) * 
            tracker_ptr->n_update_iterations() + update_iteration);
      }
    }
    
    // 计算结果
    tracker_ptr->CalculateResults(iteration);
    tracker_ptr->VisualizeResults(iteration);
    tracker_ptr->UpdateViewers(iteration);
    
    // 读取viewer保存的图像（包含渲染模型）
    // viewer保存的文件名格式：name_image_<index>.png（不带前导零）
    std::string color_filename = "color_viewer_image_" + std::to_string(iteration) + ".png";
    std::string depth_filename = "depth_viewer_image_" + std::to_string(iteration) + ".png";
    std::string ir_viewer_filename = "ir_viewer_image_" + std::to_string(iteration) + ".png";  // IR viewer渲染后的图像
    std::string ir_raw_filename = "realsense_ir_image_" + std::to_string(iteration) + ".png";  // 原始IR图像
    
    std::filesystem::path color_image_path = output_directory / color_filename;
    std::filesystem::path depth_image_path = output_directory / depth_filename;
    std::filesystem::path ir_viewer_image_path = output_directory / ir_viewer_filename;
    std::filesystem::path ir_raw_image_path = output_directory / ir_raw_filename;
    
    cv::Mat color_image = cv::imread(color_image_path.string());
    cv::Mat depth_image = cv::imread(depth_image_path.string());
    cv::Mat ir_viewer_image = cv::imread(ir_viewer_image_path.string());  // 包含渲染位姿的IR图像
    cv::Mat ir_raw_image = cv::imread(ir_raw_image_path.string());  // 原始IR图像
    
    // 验证IR viewer图像是否成功保存（包含渲染的位姿）
    if (ir_viewer_image.empty() && iteration % 100 == 0) {
      std::cout << "Warning: IR viewer image not found for frame " << iteration << std::endl;
    }
    
    if (!color_image.empty() && !depth_image.empty()) {
      // 调整图像大小使其一致（如果大小不同）
      cv::Size target_size = color_image.size();
      if (depth_image.size() != target_size) {
        cv::resize(depth_image, depth_image, target_size);
      }
      
      // 横向拼接图像（RGB在左，深度在右）
      cv::Mat combined_image;
      cv::hconcat(color_image, depth_image, combined_image);
      
      // 保存拼接后的图像
      std::stringstream combined_ss;
      combined_ss << "combined_image_" << std::setfill('0') << std::setw(6) << iteration << ".png";
      std::filesystem::path combined_image_path = output_directory / combined_ss.str();
      cv::imwrite(combined_image_path.string(), combined_image);
      
      // 删除单独的RGB、深度和IR图像文件（可选，节省空间）
      // std::filesystem::remove(color_image_path);
      // std::filesystem::remove(depth_image_path);
      // std::filesystem::remove(ir_image_path);
    }
    
    // 保存位姿（参考RBOT的方式）
    auto pose = body_ptr->body2world_pose();
    Eigen::Vector3f translation = pose.translation();
    const auto &rotation = pose.rotation().matrix();
    
    // 监控Z轴变化 - 只在极端情况下限制
    // 允许更大的变化范围以允许模型跟随移动（根据初始距离45cm调整）
    float current_z = translation(2);
    float z_change = std::abs(current_z - initial_distance);
    if (z_change > 0.25f) {  // 如果Z轴变化超过25cm，才强制限制
      float max_z = initial_distance + 0.25f;  // 最大70cm
      float min_z = initial_distance - 0.25f;  // 最小20cm
      translation(2) = std::max(min_z, std::min(max_z, current_z));
      pose.translation() = translation;
      body_ptr->set_body2world_pose(pose);
      if (iteration % 50 == 0) {  // 每50帧打印一次警告
        std::cout << "Warning: Z-axis change large (" << z_change << "m), corrected to " << translation(2) << "m" << std::endl;
      }
    }
    
    // 检测位姿变化 - 如果位姿变化很小，可能追踪丢失了
    m3t::Transform3fA pose_change = previous_pose.inverse() * pose;
    float translation_change = pose_change.translation().norm();
    Eigen::AngleAxisf rotation_change(pose_change.rotation());
    float rotation_change_angle = rotation_change.angle();
    
    // 如果位姿变化很小（<1mm平移，<1度旋转），可能是追踪丢失了
    bool possible_tracking_loss = (translation_change < 0.001f && rotation_change_angle < 1.0f * M_PI / 180.0f);
    
    // 输出调试信息（每100帧，监控追踪状态）
    if (iteration % 100 == 0) {
      std::cout << "Frame " << iteration << ": Z=" << translation(2) 
                << "m, translation=[" << translation(0) << ", " << translation(1) << ", " << translation(2) << "]"
                << ", pose_change=[" << translation_change << "m, " << rotation_change_angle * 180.0f / M_PI << "deg]" << std::endl;
      if (possible_tracking_loss && frames_since_last_detection > 50) {
        std::cout << "Warning: Possible tracking loss detected. Consider pressing 'r' to re-detect." << std::endl;
      }
    }
    
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    // 保存RGB坐标系下的位姿
    pose_ofs << iteration << " " << timestamp << " ";
    pose_ofs << translation(0) << " " << translation(1) << " " << translation(2) << " ";
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        pose_ofs << rotation(i, j) << " ";
      }
    }
    pose_ofs << "\n";
    pose_ofs.flush();
    
    // 计算并保存IR坐标系下的位姿
    // IR坐标系下的位姿 = color2ir_pose * RGB坐标系下的位姿
    m3t::Transform3fA pose_ir = color2ir_pose * pose;
    Eigen::Vector3f translation_ir = pose_ir.translation();
    const auto &rotation_ir = pose_ir.rotation().matrix();
    
    pose_ir_ofs << iteration << " " << timestamp << " ";
    pose_ir_ofs << translation_ir(0) << " " << translation_ir(1) << " " << translation_ir(2) << " ";
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        pose_ir_ofs << rotation_ir(i, j) << " ";
      }
    }
    pose_ir_ofs << "\n";
    pose_ir_ofs.flush();
    
    // 检查按键 - 在viewer更新后立即检查
    int key = cv::waitKey(30) & 0xFF;
    if (key == 27 || key == 'q' || key == 'Q') {  // ESC 或 Q
      tracking = false;
      std::cout << "Stopping tracking..." << std::endl;
      break;
    } else if (key == 'r' || key == 'R') {  // R键重新检测
      std::cout << "Re-detecting and re-aligning..." << std::endl;
      // 更新detector的位姿为当前body的位姿
      detector_ptr->set_link2world_pose(body_ptr->body2world_pose());
      
      // 执行重新检测
      std::set<std::string> redetect_names{optimizer_ptr->name()};
      std::set<std::string> redetected_names;
      if (tracker_ptr->DetectPoses(redetect_names, &redetected_names)) {
        std::cout << "Re-detection successful. Resuming tracking..." << std::endl;
        frames_since_last_detection = 0;
        // 重新启动模态
        for (const auto &modality_ptr : link_ptr->modality_ptrs())
          modality_ptr->StartModality(iteration, 0);
      } else {
        std::cout << "Re-detection failed. Continuing with current pose..." << std::endl;
      }
    }
    
    previous_pose = pose;
    frames_since_last_detection++;
    iteration++;
  }
  
  pose_ofs.close();
  pose_ir_ofs.close();
  std::cout << "Tracking finished. Results saved to " << output_directory << std::endl;
  std::cout << "  - RGB pose file: " << pose_file_path << std::endl;
  std::cout << "  - IR pose file: " << pose_ir_file_path << std::endl;
  
  // 自动生成视频
  std::cout << "\nGenerating video from images..." << std::endl;
  std::filesystem::path video_path = output_directory / "tracking_video.mp4";
  
  // 检查是否有拼接后的图像文件
  int image_count = 0;
  for (const auto &entry : std::filesystem::directory_iterator(output_directory)) {
    std::string filename = entry.path().filename().string();
    if (filename.find("combined_image_") == 0 && entry.path().extension() == ".png") {
      image_count++;
    }
  }
  
  if (image_count > 0) {
    std::cout << "Found " << image_count << " combined images. Creating video..." << std::endl;
    
    // 检查ffmpeg是否可用
    int check_ffmpeg = system("which ffmpeg > /dev/null 2>&1");
    if (check_ffmpeg != 0) {
      std::cerr << "Warning: ffmpeg is not installed or not in PATH." << std::endl;
      std::cerr << "Please install ffmpeg: sudo apt-get install ffmpeg" << std::endl;
      std::cerr << "Or manually create video using:" << std::endl;
      std::cerr << "  cd " << output_directory << std::endl;
      std::cerr << "  ffmpeg -framerate 30 -pattern_type glob -i 'combined_image_*.png' -c:v libx264 -pix_fmt yuv420p tracking_video.mp4" << std::endl;
    } else {
      // 使用ffmpeg生成视频（使用拼接后的图像）
      // 使用更可靠的方法：先切换到目录，然后使用glob模式
      std::string ffmpeg_cmd = "cd \"" + output_directory.string() + "\" && " +
                               "ffmpeg -y -framerate 30 -pattern_type glob -i 'combined_image_*.png' " +
                               "-c:v libx264 -pix_fmt yuv420p tracking_video.mp4 " +
                               "2>&1";
      
      std::cout << "Running ffmpeg command..." << std::endl;
      int ret = system(ffmpeg_cmd.c_str());
      
      if (ret == 0 && std::filesystem::exists(video_path)) {
        auto file_size = std::filesystem::file_size(video_path);
        std::cout << "Video created successfully: " << video_path << " (" 
                  << (file_size / 1024 / 1024) << " MB)" << std::endl;
      } else {
        std::cerr << "Warning: Failed to create video (exit code: " << ret << ")." << std::endl;
        std::cerr << "You can manually create video using:" << std::endl;
        std::cerr << "  cd " << output_directory << std::endl;
        std::cerr << "  ffmpeg -framerate 30 -pattern_type glob -i 'combined_image_*.png' -c:v libx264 -pix_fmt yuv420p tracking_video.mp4" << std::endl;
      }
    }
  } else {
    std::cout << "No combined images found to create video." << std::endl;
  }
  
  std::cout << "\nFinal output files:" << std::endl;
  std::cout << "  - Video: " << video_path << std::endl;
  std::cout << "  - RGB Poses: " << pose_file_path << " (RGB camera coordinate system)" << std::endl;
  std::cout << "  - IR Poses: " << pose_ir_file_path << " (IR camera coordinate system)" << std::endl;
  std::cout << "  - Images: " << output_directory << "/combined_image_*.png (RGB + Depth)" << std::endl;
  
  return 0;
}
