// 引入音频播放器和字幕容器
const audioPlayer = document.getElementById('audioPlayer');
const subtitleContainer = document.getElementById('subtitleContainer');
const subtitles = document.getElementById('subtitles');
const speakerNameElement = document.getElementById('speakerName');

let subtitleData = [];
let isSeeking = false;
let lastUpdateTime = 0; // 用于节流控制

fetch('text.txt')
    .then(response => response.text())
    .then(txtData => {
        let paragraphs = txtData.split('\n\n');
        if (paragraphs.length === 1) {
            paragraphs = txtData.split('\n');
        }
        paragraphs = paragraphs.map(p => p.trim()).filter(p => p.length > 0 && !/^\r$/.test(p));
        console.log('Paragraphs:', paragraphs);
        fetch('subtitles.json')
            .then(response => response.json())
            .then(data => {
                // 合并相同段落的字幕
                const mergedSubtitles = [];
                let paragraphIndex = 0;
                let currentParagraphText = paragraphs[paragraphIndex].trim();
                let currentParagraph = {
                    start: data[0].start,
                    end: data[0].end,
                    texts: [{ text: data[0].text, color: data[0].speaker.speaker_color, speaker: data[0].speaker.speaker_name, start: data[0].start, end: data[0].end }],
                };

                for (let i = 1; i < data.length; i++) {
                    const currentText = data[i].text.trim();

                    if (currentParagraphText.includes(currentText)) {
                        // 属于同一段落，合并文本
                        currentParagraph.end = data[i].end;
                        currentParagraph.texts.push({ text: data[i].text, color: data[i].speaker.speaker_color, speaker: data[i].speaker.speaker_name, start: data[i].start, end: data[i].end });
                    } else {
                        // 新段落，保存当前段落并重新开始
                        mergedSubtitles.push(currentParagraph);
                        paragraphIndex++;
                        currentParagraphText = paragraphs[paragraphIndex] ? paragraphs[paragraphIndex].trim() : '';
                        currentParagraph = {
                            start: data[i].start,
                            end: data[i].end,
                            texts: [{ text: data[i].text, color: data[i].speaker.speaker_color, speaker: data[i].speaker.speaker_name, start: data[i].start, end: data[i].end }],
                        };
                    }
                }
                mergedSubtitles.push(currentParagraph);
                console.log('mergedSubtitles:', mergedSubtitles);

                subtitleData = mergedSubtitles;

                // 将字幕渲染到页面
                subtitleData.forEach((paragraph) => {
                    const subtitleLine = document.createElement('div');
                    subtitleLine.classList.add('subtitle-line');
                    subtitleLine.setAttribute('data-start', paragraph.start);
                    subtitleLine.setAttribute('data-end', paragraph.end);

                    paragraph.texts.forEach(textPart => {
                        const span = document.createElement('span');
                        span.textContent = textPart.text;
                        span.style.color = textPart.color;
                        span.classList.add('subtitle-text');
                        span.setAttribute('data-start', textPart.start);
                        span.setAttribute('data-end', textPart.end);
                        span.setAttribute('data-speaker', textPart.speaker); // 保存说话人的名字
                        subtitleLine.appendChild(span);
                    });

                    subtitles.appendChild(subtitleLine);
                });

                // 监听音频时间更新事件，匹配当前播放时间的字幕
                audioPlayer.addEventListener('timeupdate', () => {
                    if (isSeeking) {
                        console.log('Skipping timeupdate because user is seeking...');
                        return;
                    }

                    const currentTime = audioPlayer.currentTime;
                    const now = Date.now();

                    // 节流控制，避免频繁更新（每 500ms 更新一次）
                    if (now - lastUpdateTime < 500) {
                        return;
                    }
                    lastUpdateTime = now;

                    console.log('Running timeupdate...');
                    let firstMatchFound = false;

                    // 遍历字幕，找到匹配的片段
                    subtitleData.forEach((paragraph, index) => {
                        const subtitleLine = subtitles.children[index];

                        paragraph.texts.forEach((textPart, spanIndex) => {
                            const span = subtitleLine.children[spanIndex];
                            const start = parseFloat(textPart.start);
                            const end = parseFloat(textPart.end);

                            // 检查是否在当前播放时间内
                            if (currentTime >= start && currentTime <= end) {
                                if (!span.classList.contains('subtitle-active')) {
                                    console.log(`Highlighting span: ${textPart.text} at time ${currentTime}`);
                                    span.classList.add('subtitle-active');
                                    speakerNameElement.textContent = textPart.speaker; // 更新说话人

                                    // 只滚动第一次找到的匹配
                                    if (!firstMatchFound) {
                                        console.log('Scrolling into view for first match found...');
                                        span.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                        firstMatchFound = true;
                                    }
                                }
                            } else {
                                if (span.classList.contains('subtitle-active')) {
                                    console.log(`Removing highlight from span: ${textPart.text}`);
                                    span.classList.remove('subtitle-active');
                                }
                            }
                        });
                    });
                });

                // 当用户开始拖动进度条时，设置 isSeeking 为 true
                audioPlayer.addEventListener('seeking', () => {
                    console.log('User started seeking...');
                    isSeeking = true;
                });

                // 当用户结束拖动进度条时，设置 isSeeking 为 false
                audioPlayer.addEventListener('seeked', () => {
                    console.log('User finished seeking...');
                    isSeeking = false;

                    // 拖动后立即更新字幕显示状态
                    const currentTime = audioPlayer.currentTime;
                    console.log(`Updating subtitles after seeking to time: ${currentTime}`);
                    let firstMatchFound = false;

                    // 遍历字幕，找到匹配的片段
                    subtitleData.forEach((paragraph, index) => {
                        const subtitleLine = subtitles.children[index];
                        paragraph.texts.forEach((textPart, spanIndex) => {
                            const span = subtitleLine.children[spanIndex];
                            const start = parseFloat(textPart.start);
                            const end = parseFloat(textPart.end);
                            if (currentTime >= start && currentTime <= end) {
                                console.log(`Highlighting span after seek: ${textPart.text}`);
                                span.classList.add('subtitle-active');
                                speakerNameElement.textContent = textPart.speaker; // 更新说话人

                                // 只滚动第一次找到的匹配
                                if (!firstMatchFound) {
                                    console.log('Scrolling into view after seek...');
                                    span.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                    firstMatchFound = true;
                                }
                            } else {
                                if (span.classList.contains('subtitle-active')) {
                                    console.log(`Removing highlight from span after seek: ${textPart.text}`);
                                    span.classList.remove('subtitle-active');
                                }
                            }
                        });
                    });
                });
            })
            .catch(error => console.error('Error loading subtitles:', error));
    })
    .catch(error => console.error('Error loading story text:', error));