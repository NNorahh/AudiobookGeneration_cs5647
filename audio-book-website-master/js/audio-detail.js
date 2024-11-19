// 引入音频播放器和字幕容器
const audioPlayer = document.getElementById('audioPlayer');
const subtitleContainer = document.getElementById('subtitleContainer');
const subtitles = document.getElementById('subtitles');
const speakerNameElement = document.getElementById('speakerName');


let subtitleData = [];
let lastUpdateTime = 0; // 用于节流控制

// 根据换行符划分段落
function splitParagraphs(text) {
    return text.split(/\n+/).map(p => p.trim()).filter(p => p.length > 0);
}

// 加载字幕json数据， 结果保存到subtitleData
function loadSubtitleData(novalName, chapterId, paragraphs) {
    return fetch(`./data/${novalName}/json/${chapterId}.json`)
        .then(response => response.json())
        .then(data => {
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
        }).catch(error => console.error('Error loading subtitles:', error));
}

// 创建字幕的 Span 元素
function createSubtitleSpan(textPart) {
    const span = document.createElement('span');
    span.textContent = textPart.text;
    span.style.color = textPart.color;
    span.classList.add('subtitle-text');
    span.setAttribute('data-start', textPart.start);
    span.setAttribute('data-end', textPart.end);
    span.setAttribute('data-speaker', textPart.speaker); // 保存说话人的名字
    return span;
}

// 将字幕渲染到页面
function renderSubtitles(subtitleData) {
    const fragment = document.createDocumentFragment(); // 使用 DocumentFragment 来提升性能

    subtitleData.forEach((paragraph) => {
        const subtitleLine = document.createElement('div');
        subtitleLine.classList.add('subtitle-line');
        subtitleLine.setAttribute('data-start', paragraph.start);
        subtitleLine.setAttribute('data-end', paragraph.end);

        paragraph.texts.forEach(textPart => {
            const span = createSubtitleSpan(textPart);
            subtitleLine.appendChild(span);
        });

        fragment.appendChild(subtitleLine);
        // 在每个段落后添加换行符
        const breakLine = document.createElement('br');
        fragment.appendChild(breakLine);
    });

    // 将整个片段一次性添加到字幕容器中
    const subtitles = document.getElementById('subtitles'); // 假设字幕容器的 ID 是 "subtitles"
    if (subtitles) {
        subtitles.appendChild(fragment);
    } else {
        console.error('字幕容器未找到，请检查 DOM 中是否存在对应的元素。');
    }
}

// 匹配当前播放时间的字幕并滚动高亮
function matchSubtitleToTime(currentTime) {
    let firstMatchFound = false;
    // 遍历字幕，找到匹配的片段
    subtitleData.forEach((paragraph, index) => {
        const subtitleLine = subtitles.children[index * 2]; // 注意要考虑 <br> 元素的存在，所以使用 index * 2

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


                    document.querySelectorAll('.character-list p').forEach(characterItem => {
                        characterItem.classList.remove('character-active');
                    });
                    // 查找对应说话人并添加高亮背景
                    const matchingCharacter = Array.from(document.querySelectorAll('.character-list p')).find(characterItem =>
                        characterItem.textContent.trim() === textPart.speaker
                    );
                    if (matchingCharacter) {
                        matchingCharacter.classList.add('character-active'); // 为匹配的角色添加高亮
                    }

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
}


function fetchTextData(url) {
    return fetch(url)
        .then(response => response.text())
        .then(txtData => splitParagraphs(txtData))
        .catch(error => {
            console.error('Error fetching text data:', error);
            throw error;
        });
}

// 主体函数
document.addEventListener('DOMContentLoaded', () => {
    fetch(`./data/AliceInWonderland/characters/1.json`)
        .then(response => response.json())
        .then(characters => {
            const characterList = document.querySelector('.character-list');
            characterList.innerHTML = ''; // 清空现有内容
            characters.forEach(character => {
                const characterItem = document.createElement('p');
                characterItem.textContent = character.character_name;
                characterItem.style.color = character.character_color;
                characterList.appendChild(characterItem);
            });
        })
        .catch(error => console.error('Error loading characters:', error));

    fetchTextData('./data/AliceInWonderland/chapters/1.txt')
        .then(paragraphs => {
            console.log('Paragraphs:', paragraphs);
            loadSubtitleData('AliceInWonderland', 1, paragraphs)
                .then(() => {
                    renderSubtitles(subtitleData);
                })
                .catch(error => console.error('Error loading subtitles:', error));

            // 监听音频时间更新事件，匹配当前播放时间的字幕
            window.audioPlayer.addEventListener('timeupdate', () => {
                const currentTime = window.audioPlayer.currentTime;
                const now = Date.now();
                // 节流控制，避免频繁更新（每 10ms 更新一次）
                if (now - lastUpdateTime < 10) {
                    return;
                }
                lastUpdateTime = now;
                console.log('Running timeupdate...');
                matchSubtitleToTime(currentTime);
            });
        })
        .catch(error => console.error('Error loading subtitles:', error));
});

function chapterChange(data_id) {
    const newAudioSrc = `../audios/chapter${data_id}.mp3`;
    updateSongSrc(newAudioSrc);
    subtitles.innerHTML = '';

    fetch(`./data/AliceInWonderland/characters/${data_id}.json`)
        .then(response => response.json())
        .then(characters => {
            const characterList = document.querySelector('.character-list');
            characterList.innerHTML = ''; // 清空现有内容
            characters.forEach(character => {
                const characterItem = document.createElement('p');
                characterItem.textContent = character.character_name;
                characterItem.style.color = character.character_color;
                characterList.appendChild(characterItem);
            });
        })
        .catch(error => console.error('Error loading characters:', error));


    fetchTextData(`./data/AliceInWonderland/chapters/${data_id}.txt`)
        .then(paragraphs => {
            console.log('Paragraphs:', paragraphs);
            loadSubtitleData('AliceInWonderland', data_id, paragraphs)
                .then(() => {
                    renderSubtitles(subtitleData);
                })
                .catch(error => console.error('Error loading subtitles:', error));
            window.audioPlayer.addEventListener('timeupdate', () => {
                const currentTime = window.audioPlayer.currentTime;
                const now = Date.now();
                // 节流控制，避免频繁更新（每 10ms 更新一次）
                if (now - lastUpdateTime < 10) {
                    return;
                }
                lastUpdateTime = now;
                console.log('Running timeupdate...');
                matchSubtitleToTime(currentTime);
            });
        })
        .catch(error => console.error('Error loading subtitles:', error));
    // fetch(`./data/AliceInWonderland/json/${data_id}.txt`)
    //     .then(response => response.text())
    //     .then(txtData => {
    //         let paragraphs = splitParagraphs(txtData);
    //         console.log('Paragraphs:', paragraphs);
    //         loadSubtitleData('AliceInWonderland', data_id, paragraphs)
    //             .then(() => {
    //                 renderSubtitles(subtitleData);
    //             })
    //             .catch(error => console.error('Error loading subtitles:', error));
    //         // 监听音频时间更新事件，匹配当前播放时间的字幕
    //         window.audioPlayer.addEventListener('timeupdate', () => {
    //             const currentTime = window.audioPlayer.currentTime;
    //             const now = Date.now();
    //             // 节流控制，避免频繁更新（每 10ms 更新一次）
    //             if (now - lastUpdateTime < 10) {
    //                 return;
    //             }
    //             lastUpdateTime = now;
    //             console.log('Running timeupdate...');
    //             matchSubtitleToTime(currentTime);
    //         });
    //     })
    //     .catch(error => console.error('Error loading subtitles:', error));
}

// 示例：点击某个项目时，加载内容
document.querySelectorAll('.audio-top-item').forEach(item => {
    item.addEventListener('click', event => {
        event.preventDefault();
        let itemId = item.dataset.id;
        chapterChange(itemId); // 加载对应的文本数据
    });
});

function updateSongSrc(newSrc) {
    // 暂停当前播放
    window.audioPlayer.pause();

    // 更新音频源
    window.audioPlayer.src = newSrc;
    window.audioPlayer.load(); // 重新加载音频文件
    window.audioPlayer.play(); // 自动播放

    // 更新播放按钮图标为暂停
    $("#play img").attr("src", "./img/pause-button.png");
}