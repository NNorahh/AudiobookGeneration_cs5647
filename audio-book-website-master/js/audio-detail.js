// 引入音频播放器和字幕容器
const audioPlayer = document.getElementById('audioPlayer');
const subtitleContainer = document.getElementById('subtitleContainer');
const subtitles = document.getElementById('subtitles');
const speakerNameElement = document.getElementById('speakerName');

let subtitleData = [];

fetch('subtitles.json')
    .then(response => response.json())
    .then(data => {
        subtitleData = data;
        // 将字幕渲染到页面
        subtitleData.forEach((line, index) => {
            const subtitleLine = document.createElement('div');
            subtitleLine.classList.add('subtitle-line');
            subtitleLine.setAttribute('data-start', line.start);
            subtitleLine.setAttribute('data-end', line.end);
            subtitleLine.textContent = line.text;
            subtitleLine.style.color = line.speaker.speaker_color; // 设置角色颜色
            subtitles.appendChild(subtitleLine);
        });
        // 监听音频时间更新事件，匹配当前播放时间的字幕
        audioPlayer.addEventListener('timeupdate', () => {
            const currentTime = audioPlayer.currentTime;

            // 遍历字幕，找到匹配的片段
            subtitleData.forEach((line, index) => {
                const start = line.start;
                const end = line.end;
                const subtitleLine = subtitles.children[index];

                // 检查是否在当前播放时间内
                if (currentTime >= start && currentTime <= end) {
                    subtitleLine.classList.add('subtitle-active');
                    subtitleLine.style.color = line.speaker.speaker_color;
                    // 滑动到当前字幕
                    subtitleLine.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    speakerNameElement.textContent = `${line.speaker.speaker_name}`;
                } else {
                    subtitleLine.classList.remove('subtitle-active');
                    subtitleLine.style.color = 'initial';
                }
            });
        });
    })
    .catch(error => console.error('Error loading subtitles:', error));
// // 加载字幕文件
// const subtitleData = [
//     { "start": 0, "end": 4, "text": "This is the beginning of the story." },
//     { "start": 5, "end": 9, "text": "It continues with an interesting event." },
//     // 添加更多字幕片段...
// ];