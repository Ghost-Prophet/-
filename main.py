import os
import json
import time
import base64
from openai import OpenAI

def main():
    # 配置
    API_KEY = "sk-ffdaec9da0eb4024bface0c5a53cedb3"
    IMAGE_DIR = "images"  # 图片目录
    OUTPUT_DIR = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 系统提示（用于第二阶段：任务执行）
    system_prompt_for_planning = """# Role
You are an advanced underwater robot vision-language-action (VLA) task planning model. Your core capability is to decompose complex macroscopic tasks into logically rigorous, executable natural language atomic step sequences based on monocular RGB images (visual input) and user natural language instructions.

# Context & Constraints
1. **Environment**: Underwater environment with potential uneven lighting, turbidity, or dynamic obstacles.
2. **Input**:
    - Monocular RGB image from current perspective (analyze object positions, orientations, and obstacles).
    - User macroscopic instruction (e.g., "Avoid red obstacles and turn right" or "Approach the side rock").
3. **Output Requirements**:
    - Must conform to provided JSON Schema format.
    - Action field must be a list of strings.
    - Step descriptions should be concise, natural, and action-oriented (e.g., "Turn right", "Go straight").

# Workflow
Before generating output, perform these internal reasoning steps:
1. Visual Perception: Identify key targets and obstacles in the image.
2. Spatial Reasoning: Determine relative positions (left? front? distance?) between robot and target.
3. Plan Decomposition: Break down path into sequential actions based on spatial relationships.

# Task
Generate corresponding reasoning process and action sequence based on provided image and instruction."""

    # 系统提示（用于第一阶段：任务提议）
    system_prompt_for_task_proposal = """You are an intelligent underwater robot observer. Based solely on the provided underwater monocular RGB image, propose a single, reasonable, and actionable high-level task that the robot could perform in this scene.

The task should:
- Be specific and grounded in visible objects (e.g., rocks, corals, pipes, obstacles).
- Use natural language (e.g., "Approach the yellow sponge", "Avoid the red buoy and go forward").
- Not be vague (avoid "do something", "explore", etc.).
- Be feasible for a mobile robot to execute.

Respond ONLY with the task description as a plain string. Do not add explanations, markdown, or quotes."""

    # 获取图片目录中的所有图片
    image_files = [f for f in os.listdir(IMAGE_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"❌ No images found in {IMAGE_DIR}. Please add images to the images/ directory.")
        return

    print(f"🔍 Found {len(image_files)} images in {IMAGE_DIR}. Processing all images...")
    
    # 处理每张图片
    for image_file in image_files:
        image_path = os.path.join(IMAGE_DIR, image_file)
        print(f"\n{'='*50}")
        print(f"Processing image: {image_path}")
        print(f"{'='*50}")
        
        # 读取图片并编码
        try:
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            print(f"  ❌ Error reading image: {str(e)}")
            continue
        
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # ===== 第一阶段：让模型提出任务 =====
        print("  🧠 Stage 1: Generating task proposal...")
        messages_proposal = [
            {"role": "system", "content": system_prompt_for_task_proposal},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ]
        
        try:
            response_proposal = client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=messages_proposal,
                temperature=0.4,
                max_tokens=100
            )
            proposed_task = response_proposal.choices[0].message.content.strip()
            print(f"  💡 Proposed task: {proposed_task}")
        except Exception as e:
            print(f"  ❌ Failed to generate task: {str(e)}")
            proposed_task = "No task could be generated."
        
        # ===== 第二阶段：基于该任务生成 plan =====
        print("  🤖 Stage 2: Generating action plan based on proposed task...")
        messages_planning = [
            {"role": "system", "content": system_prompt_for_planning},
            {"role": "user", "content": [
                {"type": "text", "text": proposed_task},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ]
        
        try:
            response_planning = client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=messages_planning,
                temperature=0.4, #low temperature
                max_tokens=2000
            )
            planning_text = response_planning.choices[0].message.content
            
            # 尝试解析为 JSON
            try:
                output_json = json.loads(planning_text)
            except json.JSONDecodeError:
                # 如果不是合法 JSON，包装成默认结构
                output_json = {
                    "reasoning": "Failed to parse model output as JSON.",
                    "actions": []
                }
                print(f"  ⚠️ Warning: Model output is not valid JSON. Raw output:\n{planning_text[:200]}...")
                
        except Exception as e:
            print(f"  ❌ Failed to generate plan: {str(e)}")
            output_json = {
                "reasoning": f"Error during planning: {str(e)}",
                "actions": []
            }
        
        # 构建最终结果
        result = {
            "instruction": system_prompt_for_planning,  # 可选：也可存两个 system prompt
            "input": proposed_task,                     # ← 这是模型自己提出的任务
            "output": output_json,
            "images": [image_path]
        }
        
        # 保存结果
        output_file = os.path.join(OUTPUT_DIR, f"result_{os.path.splitext(image_file)[0]}.json")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"  ✅ Saved to: {output_file}")
        print(f"  Example reasoning: {output_json.get('reasoning', '')[:60]}...")

        # 避免请求过快
        time.sleep(1.5)
        
    print(f"\n{'='*50}")
    print(f"✅ All {len(image_files)} images processed! Results saved to {OUTPUT_DIR}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()