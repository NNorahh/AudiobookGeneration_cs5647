import gradio as gr

css = """
.gradio-container.gradio-container-5-3-0.svelte-18jtcab.app{
    /*background-image:url('https://images.unsplash.com/photo-1577563908411-5077b6dc7624?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb&dl=volodymyr-hryshchenko-V5vqWC9gyEU-unsplash.jpg&w=2400'); */
    /*https://images.pexels.com/photos/673535/pexels-photo-673535.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2background-image:url('https://images.pexels.com/photos/673535/pexels-photo-673535.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');*/
    background-image:url('https://mail.google.com/mail/u/1?ui=2&ik=bbd95ff9e8&attid=0.1&permmsgid=msg-a:r528975396292630947&th=192e2afeee532e20&view=fimg&fur=ip&sz=s0-l75-ft&attbid=ANGjdJ9vEcJCGip49kYJEjqRCwH_E-sscM6DBr9R0hqJn_HCByRoeeqTTvplm4hLtmk4cjb0fDzpV3HFhhKmAxcjUFPr0jVH3uyvNuywcHl9Tko1fhrw_HUhCdg-q6w&disp=emb&realattid=ii_m2xblcar0');
    background-size:cover; /*整张图片覆盖页面*/
    background-position: center;
    background-repeat:repeat;
}

.backgroundcolor {
    /* background-image:url('https://mail.google.com/mail/u/1?ui=2&ik=bbd95ff9e8&attid=0.1&permmsgid=msg-a:r-5179234301891548155&th=192baf110f31816f&view=fimg&fur=ip&sz=s0-l75-ft&attbid=ANGjdJ9OdjggRq5JQBSnHjsTZKDTsnbkK_ZITu-TJe07Umc6_1OTmroImMxjFHUrXO0w8339h8llyBKjBbl0bUtE7afo949rGoL0le8yvBVnns_0TDHQaIjnebVkIfk&disp=emb&realattid=ii_m2mal1840');*/
    background-size: cover      /* 保持比例，覆盖整个容器 https://images.pexels.com/photos/9743265/pexels-photo-9743265.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2*/
    /*background-position: center;  背景图片居中    */
    background-repeat: no-repeat; /* 不重复显示背景 */
    background-color: grey;    /*设置背景颜色 */
    /*border-radius: 5px;  圆角边框 */
}

.blockcolor {
    background-color: lightgrey;
    /*background-image:url('https://images.pexels.com/photos/1037995/pexels-photo-1037995.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');*/
    background-size: cover;      /* 保持比例，覆盖整个容器 */
    background-position: center; /* 背景图片居中 */
    background-repeat: no-repeat; /* 不重复显示背景 */
    /*border-radius: 5px;  圆角 */
}

#custom-group {
    border: none;   /*设置边框颜色 */
    padding:10px;
    min-height: 800px; /* 最小高度为 800px */
    height: auto;    /* 高度自动调整以适应内容 */
    overflow: visible; /* 确保内容不会被裁剪 */
}

#file-upload {
    height:250px;
    margin-bottom: 20px;
    border:none;
}  

#button1 {
    height:70px;
    border-radius: 5px; /* 圆角 */
    margin-bottom:20px;
    background-color: #f0f8ff;
    font-size: 20px; 
    border:none;
}

#audio1{
    height:250px;
    margin-bottom:20px;
}

#heading {
    /* padding-left:150px; */
    /* padding-right:150px;*/
    padding-top:10px;
    padding-bottom:10px;
    text-align:center;
    border:none;
    color:white;
    height: 50px;
    /*background-image:url('https://images.pexels.com/photos/9743265/pexels-photo-9743265.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'); */
    background-color: lightgrey;
    background-size: cover;      /* 保持比例，覆盖整个容器 */
}

.gr-tabs {
    display: flex;                /* 使用 Flexbox 布局 */
    /*justify-content: space-between;  横向均匀分布 */
    height: auto;                 /* 高度自适应 */
    /*padding: 5px;                内边距 */
    overflow: hidden;            /* 隐藏超出部分 */
    /*background-image:url('https://mail.google.com/mail/u/1?ui=2&ik=bbd95ff9e8&attid=0.1&permmsgid=msg-a:r-5179234301891548155&th=192baf110f31816f&view=fimg&fur=ip&sz=s0-l75-ft&attbid=ANGjdJ9OdjggRq5JQBSnHjsTZKDTsnbkK_ZITu-TJe07Umc6_1OTmroImMxjFHUrXO0w8339h8llyBKjBbl0bUtE7afo949rGoL0le8yvBVnns_0TDHQaIjnebVkIfk&disp=emb&realattid=ii_m2mal1840');  */
    background-color: grey;    /*设置 tabs 区域的背景颜色 */
    /*border-radius: 10px;           设置圆角边框 */
}

/* 按钮样式 */
.gr-tabs button {
    /*background-image:url('https://images.pexels.com/photos/2847646/pexels-photo-2847646.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');*/
    background-color: white;  /*#CCCCFF;按钮背景颜色 */
    height: auto;
    color: black;                 /* 按钮字体颜色 */
    border-bottom: 2px solid #CCCCFF; 
    /*padding: 10px;                  调整内边距 */
    font-size: 20px;              /* 字体大小 */
    flex: 1;                      /* 使按钮平均分配空间 */
    margin-right: 30px;                 /* 外边距 */
    margin-left:5px;
    margin-top:5px;
    /* box-sizing: border-box;        包括内边距和边框在内的宽度计算 */
}

/* 悬停和焦点样式 */
.gr-tabs button:hover {
    background-color: lightgrey; /* 悬停时背景颜色 */
    border-bottom: 2px solid #CCCCFF; 
}

.gr-tabs button:focus, .gr-tabs button:active {
    background-color: darkgrey; /* 选中时背景颜色 */
    color: white;              /* 选中时字体颜色 */
    /*border-bottom: 2px solid #CCCCFF; */
    outline:none;
    box-shadow: none;
}

.title{
    text-align:center;
    margin-left:300px;
    marigin-right:300px;
    padding-top:5px;
    padding-bottom:5px;
    background-color: #CCCCFF; /*white; */
    width: 200px;/**/
}

.clearcolor {
    background-color: transparent;
}

#img1 {
    background-color: transparent;  /* 确保图片背景透明 */
    display: block;
    position: absolute;  /* 使图片使用绝对定位 */
    top: 0;              /* 定位到容器顶部 */
    right: 0;            /* 定位到容器右侧 */
    background-color: white;  /* 确保图片背景透明 */
    border: none;        /* 移除边框 */
    width: 300px;        /* 调整图片大小，可以根据需求更改 */
    height: auto;        /* 使图片保持比例 */
    padding:5px;
}
"""


def read_file(file):
    if file is not None:
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    return "No file uploaded"

import openai
import os
api_key = os.getenv("apikey")
openai.api_key = api_key

prompt = (
    "1.Indentify all the speakers in the text, label them with narrator, male1, male2, female1, female2, etc. 2.Assign a corresponding speaker label and a corresponding emotion label to each sentence from front to back, consecutive sentences of the same speaker can be combined together. 3.Output in the Json format of {text: string , speaker_label: string ,emotion_label: string}"
)
SYSTEM_PROMPT = """
The user will provide a piece of a novel. Your task is the following three tasks, and you should **only return the output of the third task**: 
1. First, identify all characters  (including the narrator) and attempt to infer each character's name and gender(the gender are only used to define the voice name in the following like if the gender is women then the voice name must not be attributed to men or boy) in the novel you should list the characters like(if you are not clear about the speaker assign it to the narrator): 
Character 1: Narrator  (always say Plot Development:,Character Introductions,Inner Thoughts and so on) 
Character 2: (the name of character) 
Character 3: (the name of character) 
...(as many characters as you can define!!) 
do not return the list. 
subtask2:
go through all the sentences in the novel, return every sentence(consider "," also as the end of a sentence) in the novel(output one sentence every time), voice name, color(the color should be bind to the character name(the inferred character from the character list by you),which means one sentence has one character bind with unique voice name and color) and a reference number (starting from 1, with each number referring to a sentence or narration by the character) like: 
**Voice Names **: Convert the novel text into a script format by inferring one voice name from the options below(distinguish each voice based on their description): 
                    - alloy 
                    - echo 
                    - fable 
                    - onyx 
                    - nova 
 				    - shimmer 
     description: 
                    -alloy:narrator's voice 
                    -echo:young men adult 
                    -fable:teenager boy 
                    -onyx:mid-aged men 
                    -nova:mid-aged women 
                    -shimmer:teenager girl 
Important note:one voice name can only be attributed to one.
character,Alice should use 'shimmer'. 
subtask3:
Then, transform the novel into a script format. Any introductory titles or author information should be treated as part of the narrator’s lines.  
                   Format 
                Each response line should follow this JSON format:(begin directly with"[" and end with "]") 
                -"sentence": "string"(sentence in the novel), 
                    -"reference number": int, 
                    -"character name": "string", 
                    -"voice name": "string", 
                    -"color": "string"，
"""

def process_with_chatgpt(file_content):
    try:
        # 调用 OpenAI ChatGPT API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{file_content}"}
            ]
        )
        result = response.choices[0].message.content
        return result
    except Exception as e:
        return f"Error with ChatGPT API: {str(e)}"

from pydub import AudioSegment

def handle_file(file):
    file_content = read_file(file)
    if file_content.startswith("No file uploaded"):
        return file_content, None  

    result = process_with_chatgpt(file_content)
    
    try:
        sentences = eval(result)  

        temp_audio_folder = "temp_audio"
        os.makedirs(temp_audio_folder, exist_ok=True)

        audio_files = []
        
        for entry in sentences:
            sentence = entry["sentence"]
            voice_name = entry["voice name"]
            
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice_name,
                input=sentence
            )

            audio_data = response.content  
            audio_path = os.path.join(temp_audio_folder, f"{entry['reference number']}.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            audio_files.append(audio_path)

        final_audio = AudioSegment.empty()
        for audio_file in sorted(audio_files, key=lambda x: int(os.path.basename(x).split('.')[0])):
            segment = AudioSegment.from_file(audio_file)
            final_audio += segment

        output_audio_path = "final_audio.mp3"
        final_audio.export(output_audio_path, format="mp3")

        return result, output_audio_path

    except Exception as e:
        return f"Error processing audio: {str(e)}", None

    
html_content = """
<img src="https://www.comp.nus.edu.sg/wp-content/uploads/2023/09/nus-computing-logo-1.png" alt="Example Image">
"""

html_code = """
<div style="width: 100%; text-align: center; position: relative; top: 0;">
        <h1 style="color: white; padding-left: 10px;padding-right: 10px;padding-bottom: 12px;">Generate Your Own Audiobook</h1>
    
</div>
"""
with gr.Blocks(css=css) as demo:
    gr.HTML(html_content, elem_id="img1")
    # gr.Image("./nus-computing-logo-1.png",elem_id="img1")
    with gr.Row():
        # gr.HTML(html_content)
        #gr.HTML("<h1>Generate Your Own Audiobook</h1>",elem_classes="title")
        gr.HTML(html_code)
            
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Group(elem_id="custom-group", elem_classes="backgroundcolor"):
                    gr.HTML("<h2> Input The Text Here</h2>", elem_id="heading")
                    with gr.Tabs(elem_classes="gr-tabs"):
                        with gr.Tab(label="Upload Your Text Here", elem_classes="backgroundcolor"):
                            file_input = gr.File(label="Upload Text File", file_types=['.txt'], interactive=True,
                                                 elem_id="file-upload", elem_classes="blockcolor")
                            btn = gr.Button("Read File", elem_id="button1")
                            file_output = gr.Textbox(label="File Content", placeholder="Here is your file content", lines=12, elem_classes="blockcolor")
                            btn.click(read_file, inputs=file_input, outputs=file_output)

                        #with gr.Tab(label="Type Your Text Here", elem_classes="backgroundcolor"):
                        #    textbox = gr.TextArea(label="Input Text",
                        #                          placeholder="Paste or type the text you want to synthesize here.",
                        #                          lines=15,
                        #                          elem_classes="blockcolor")

        with gr.Column():
            with gr.Row():
                with gr.Group(elem_id="custom-group", elem_classes="backgroundcolor"):
                    gr.HTML("<h2>Generate The Audio Here</h2>", elem_id="heading")
                    with gr.Tabs(elem_classes="gr-tabs"):
                        with gr.Tab(label="Audio Output", elem_classes="backgroundcolor"):
                            audio_output = gr.Audio(label="Output Audio", interactive=False, elem_id="audio1", elem_classes="blockcolor")
                            btn_generate = gr.Button("Generate!", elem_id="button1")
                            

                    gr.HTML("<h2>Interim Output</h2>", elem_id="heading")

                    with gr.Tabs(elem_classes="gr-tabs"):
                        with gr.Tab(label="Classification Output", elem_classes="backgroundcolor"):
                            cls_output = gr.TextArea(label="Classification Output", placeholder="Here is your classification output",
                                                  lines=7,
                                                  elem_classes="blockcolor")

                    btn_generate.click(handle_file, inputs=file_input, outputs=[cls_output, audio_output])

demo.launch()