import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import math
import tempfile
import pandas as pd
from datetime import datetime

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="YOLO11 + 平滑人数折线图", layout="wide")
st.title("🚀 YOLO11 视频/图片检测 | 人数实时平滑折线图（时间轴）")

# -------------------------- 全局状态 --------------------------
if "prev_keypoints" not in st.session_state:
    st.session_state.prev_keypoints = None
if "motion_scores" not in st.session_state:
    st.session_state.motion_scores = []
# 历史数据（原始人数）
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["时间", "人数"])

# 平滑窗口大小
SMOOTH_WIN = 10


# -------------------------- 加载模型 --------------------------
@st.cache_resource
def load_model(task_type="detect"):
    if task_type == "pose":
        return YOLO("yolo11s-pose.pt")
    else:
        return YOLO("yolo11s.pt")


# -------------------------- 运动强度计算 --------------------------
def calculate_motion_intensity(current_kps, prev_kps, threshold=10):
    if current_kps is None or prev_kps is None:
        return 0.0
    total_dist, count = 0.0, 0
    for i in range(len(current_kps)):
        x1, y1, conf1 = current_kps[i]
        x2, y2, conf2 = prev_kps[i]
        if conf1 > 0.5 and conf2 > 0.5:
            dist = math.hypot(x1 - x2, y1 - y2)
            total_dist += dist
            count += 1
    if count == 0:
        return 0.0
    return max(0.0, (total_dist / count) - threshold)


# -------------------------- 活动强度分级 --------------------------
def get_activity_level(score):
    if score < 3:
        return "极低 🟢"
    elif score < 8:
        return "低 🟡"
    elif score < 18:
        return "中等 🟠"
    elif score < 35:
        return "高 🔴"
    else:
        return "剧烈 🔥"


# -------------------------- 绘制骨架 --------------------------
def draw_skeleton(img, kps):
    skeleton = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
                (14, 16)]
    for (p1, p2) in skeleton:
        if kps[p1][2] > 0.5 and kps[p2][2] > 0.5:
            cv2.line(img, tuple(kps[p1][:2].astype(int)), tuple(kps[p2][:2].astype(int)), (0, 255, 0), 2)
    for i in range(17):
        if kps[i][2] > 0.5:
            cv2.circle(img, tuple(kps[i][:2].astype(int)), 3, (0, 0, 255), -1)
    return img


# -------------------------- 侧边栏参数 --------------------------
st.sidebar.header("⚙️ 参数设置")
task = st.sidebar.selectbox("选择任务", ["目标检测", "姿态估计"])
conf_thresh = st.sidebar.slider("置信度阈值", 0.1, 0.9, 0.25)
smooth_win = st.sidebar.slider("平滑窗口帧数", 3, 20, SMOOTH_WIN)

task_map = {"目标检测": "detect", "姿态估计": "pose"}
model = load_model(task_map[task])

# -------------------------- 图片/视频选择 --------------------------
mode = st.radio("选择模式", ["图片", "视频"])

if mode == "图片":
    uploaded_img = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        img_np = np.array(img)
        result = model.predict(source=img_np, save=False, verbose=False)[0]
        annotated_img = result.plot()

        person_count = 0
        if task == "姿态估计" and result.keypoints is not None:
            all_kps = result.keypoints.cpu().numpy()
            person_count = len(all_kps)
            for one_kp in all_kps:
                current_kps = np.concatenate([one_kp[:, :2], one_kp[:, 2:]], axis=1)
                annotated_img = draw_skeleton(annotated_img, current_kps)
        else:
            if result.boxes is not None:
                cls_ids = result.boxes.cls.cpu().numpy()
                person_count = sum(1 for cls in cls_ids if int(cls) == 0)

        st.info(f"图片检测总人数：{person_count}")
        st.image(annotated_img, caption="推理结果", use_container_width=True)

else:
    uploaded_vid = st.file_uploader("上传视频", type=["mp4", "mov", "avi"])
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 左画面，右平滑折线图
        col1, col2 = st.columns(2)
        st_frame = col1.empty()
        line_chart_place = col2.empty()
        info_place = st.empty()

        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cnt += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = model.predict(frame_rgb, conf=conf_thresh, verbose=False)[0]
            ann_frame = res.plot()

            motion_score = 0.0
            smooth_score = 0.0
            level = "无数据"
            person_count = 0

            if task == "姿态估计" and res.keypoints is not None and len(res.keypoints) > 0:
                all_person_kps = res.keypoints.cpu().numpy()
                person_count = len(all_person_kps)
                first_kps = np.concatenate([all_person_kps[0][:, :2], all_person_kps[0][:, 2:]], axis=1)

                for per_kp in all_person_kps:
                    single_kp = np.concatenate([per_kp[:, :2], per_kp[:, 2:]], axis=1)
                    ann_frame = draw_skeleton(ann_frame, single_kp)

                if st.session_state.prev_keypoints is not None:
                    motion_score = calculate_motion_intensity(first_kps, st.session_state.prev_keypoints)
                st.session_state.prev_keypoints = first_kps

                st.session_state.motion_scores.append(motion_score)
                if len(st.session_state.motion_scores) > 15:
                    st.session_state.motion_scores.pop(0)
                smooth_score = np.mean(st.session_state.motion_scores)
                level = get_activity_level(smooth_score)

                cv2.putText(ann_frame, f"Activity: {level}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(ann_frame, f"Score: {smooth_score:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 255), 3)
            else:
                if res.boxes is not None:
                    cls_ids = res.boxes.cls.cpu().numpy()
                    person_count = sum(1 for cls in cls_ids if int(cls) == 0)

            # 1. 存入原始数据
            now_str = datetime.now().strftime("%H:%M:%S")
            new_row = pd.DataFrame([[now_str, person_count]], columns=["时间", "人数"])
            st.session_state.history_df = pd.concat([st.session_state.history_df, new_row], ignore_index=True)

            # 限制总长度，防止内存占用过高
            if len(st.session_state.history_df) > 500:
                st.session_state.history_df = st.session_state.history_df.iloc[-500:]

            # 2. 计算平滑数据（不额外创建 DataFrame，直接在原始数据上计算）
            st.session_state.history_df["平滑人数"] = st.session_state.history_df["人数"].rolling(window=smooth_win,
                                                                                                  min_periods=1).mean()

            # 3. 绘制平滑后的折线图
            line_chart_place.line_chart(
                st.session_state.history_df,
                x="时间", y="平滑人数",
                use_container_width=True
            )

            st_frame.image(ann_frame, caption="视频推理", use_container_width=True)
            info_place.markdown(
                f"**帧：{cnt}/{total_frames} | 当前原始人数：{person_count}**<br>"
                f"**运动得分：{smooth_score:.1f} | 活动强度：{level} | 置信度：{conf_thresh}**"
            )

        cap.release()
        st.success("✅ 视频处理完成")