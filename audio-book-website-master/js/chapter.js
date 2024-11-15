function toggleLeftside() {
    const leftside = document.getElementById("leftside");
    const main = document.getElementById("main");

    // 检查 rightside 是否显示，进行相反的操作
    if (leftside.style.display === "none" || leftside.style.display === "") {
        leftside.style.display = "block"; // 显示 rightside
        main.style.gridColumnStart = "2"; // 切换到第二列
        main.style.gridColumnEnd = "5";
    } else {
        leftside.style.display = "none"; // 隐藏 rightside
        main.style.gridColumnStart = "1"; // 切换到第二列
        main.style.gridColumnEnd = "5";
    }
}

//在项目根目录打开终端并运行 python -m http.server
//访问 http://localhost:8000 以查看项目。

/*
function loadTextData(itemId) {
    fetch('../data/textData.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // 清空现有内容
            const textContainer = document.getElementById('textContainer');
            textContainer.innerHTML = '';

            // 获取 itemId 对应的句子数组
            const sentences = data[itemId];

            // 将每句文本创建为带有颜色的 `<p>` 元素
            sentences.forEach(sentence => {
                const p = document.createElement('p');
                p.textContent = sentence.diaglogue;
                p.style.color = sentence.color; // 应用颜色
                textContainer.appendChild(p);
            });
        })
        .catch(error => {
            console.error('Error loading text data:', error);
        });

    // 读取 JSON 数据并显示相应内容
    fetch('./data/character.json')
        .then(response => response.json())
        .then(data => {
            const audioItems = document.querySelectorAll('.audio-top-item');

            audioItems.forEach(item => {
                item.addEventListener('click', function() {
                    const id = this.getAttribute('data-id');
                    const characters = data[id];

                    // 清空 side 内容
                    const characterContainer = document.querySelector('.character');
                    characterContainer.innerHTML = '';

                    // 添加每个字符的文本和颜色
                    characters.forEach(character => {
                        const p = document.createElement('p');
                        p.textContent = character.character_name;
                        p.style.color = character.color; // 设置颜色
                        characterContainer.appendChild(p); // 将 <p> 添加到 container
                    });

                    // 显示 <aside id="side">
                    document.getElementById('side').style.display = 'block'; // 确保侧边栏可见
                });
            });
        })
        .catch(error => console.error('Error loading text data:', error));
}

// 加载角色数据的函数
function loadCharacterData(itemId) {
    fetch('./data/character.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const characters = data[itemId];

            // 清空 character 内容
            const characterContainer = document.querySelector('.character');
            characterContainer.innerHTML = '';

            // 添加每个字符的文本和颜色
            characters.forEach(character => {
                const p = document.createElement('p');
                p.textContent = character.character_name;
                p.style.color = character.color; // 设置颜色
                characterContainer.appendChild(p); // 将 <p> 添加到 container
            });

            // 显示 <aside id="side">
            document.getElementById('side').style.display = 'block'; // 确保侧边栏可见
        })
        .catch(error => console.error('Error loading character data:', error));
}
*/
