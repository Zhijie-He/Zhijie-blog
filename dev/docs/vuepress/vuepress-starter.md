# 快速部署VuePress
这里介绍了如何快速的创建一个vuepress项目并将其部署到github pages页面。
案例：https://zhijie-he.github.io/learning-blogs/

## 创建github库
因为最终要将vuepress的网站部署到github pages上面，所以第一步首先创建GitHub库，如果不打算部署到github pages上面，可以跳过此步骤。

### 创建public库
对于部署到github pages上面的库要求必须是公开的。创建库完毕后，git到本地就可以进行本地编辑。这里以learning-blogs为例。

### 文档结构
git到刚才创建的库之后，

```
cd learning-blogs
```

然后我们创建这样的目录结构
```
.
├─ dev
└─ docs
└─ README.md
└─ .gitignore
```

dev目录作为我们的主要开发目录，也就是markdown文档以及vuepress代码的存储地方。docs文档这里与vuepress中的docs文档不同， 这个docs文档作为最终部署的文档，我们也称为docs.

## 依赖环境

* Node.js
* npm / yarn (推荐使用yarn)

## 创建项目
以下命令如无特别指出，都在dev文件夹下运行：
1. 初始化项目 `yarn init`
2. 修改dev文件夹下 package.json文件中name的值为github repo的名称，即为`learning-blogs`
3. 安装vuepress `yarn add -D vuepress`
4. .gitignore文件中忽略一些不必要的文件

```
# VuePress files
docs/.vuepress/.temp/
docs/.vuepress/.cache/
docs/.vuepress/dist/

# Node modules
node_modules/

# MacOS Desktop Services Store
.DS_Store

# Log files
*.log
```

5.在package.json 中添加一些 scripts
```
{
  "scripts": {
    "docs:dev": "vuepress dev docs",
    "docs:build": "vuepress build docs"
  }
}
```
docs:dev和docs:build实际上是简化vuepress的指令。
当我们在开发阶段，我们使用docs:dev模式，当我们开发完毕发版则用docs:build。


6.创建第一篇文档
`echo '# Hello VuePress' > docs/README.md`


7. 运行
`yarn docs:dev`

VuePress 会在 <http://localhost:8080> 启动一个热重载的开发服务器。当你修改你的 Markdown 文件时，浏览器中的内容也会自动更新。

当看到这个地址下的Hello VuePress文字也就代表了vuepress初步完成。


## 基础配置
在docs文件夹下面创建一个.vuepress文件夹, 而文件夹下config.js文件则用来定义配置。
```
.
├─ docs
│  ├─ README.md
│  └─ .vuepress
│     └─ config.js
└─ package.json
```

一个 VuePress 网站必要的配置文件是 .vuepress/config.js，它应该导出一个 JavaScript 对象：
```
module.exports = {
  title: 'Hello VuePress',
  description: 'Just playing around'
}
```
对于上述的配置，如果你运行起 `yarn docs:dev`，你应该能看到一个页面，它包含一个页头，里面包含一个标题Hello VuePress和一个搜索框。VuePress 内置了基于 headers 的搜索 —— 它会自动为所有页面的标题、h2 和 h3 构建起一个简单的搜索索引。


### 侧边栏Sidebar
想要使侧边栏（Sidebar）生效，需要配置 themeConfig.sidebar，基本的配置，需要一个包含了多个链接的数组：

```
// .vuepress/config.js
module.exports = {
  themeConfig: {
    sidebar: [
      '/',
      '/page-a',
      ['/page-b', 'Explicit link text']
    ]
  }
}
```
可以省略 .md 拓展名，同时以 / 结尾的路径将会被视为 */README.md，这个链接的文字将会被自动获取到（无论你是声明为页面的第一个 header，还是明确地在 YAML front matter 中指定页面的标题）。如果你想要显示地指定链接的文字，使用一个格式为 [link, text] 的数组。

`注意：侧边栏的文字如无制定，默认是由指定页面的H1大标题决定`

## 部署到github pages
项目开发完毕后，使用`yarn docs:build`构建项目，会在`dev/docs/.vuepress/dist`文件夹下构建最终的页面，然后我们将dist文件加下的所有文件去复制到与最初dev同级目录下的docs文件。然后将文件上传到github库。

### 设置github pages
1. 打开github对应的库，并点击settings
2. 在侧边栏找到Code and automation栏目，点击Pages
3. 设置Branch信息，选择main brance并且设置文件目录为与dev文件夹同级目录的docs目录
4. 保存之后，会给出网站链接，等待刷新即部署成功。
5. 再次更新vuepress内容，只需要构建完成后，复制dist文件内容到docs文件，然后push到github，即可自动部署。