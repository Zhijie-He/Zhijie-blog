(window.webpackJsonp=window.webpackJsonp||[]).push([[53],{343:function(e,s,a){"use strict";a.r(s);var t=a(14),r=Object(t.a)({},(function(){var e=this,s=e._self._c;return s("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[s("h1",{attrs:{id:"快速部署vuepress"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#快速部署vuepress"}},[e._v("#")]),e._v(" 快速部署VuePress")]),e._v(" "),s("p",[e._v("这里介绍了如何快速的创建一个vuepress项目并将其部署到github pages页面。")]),e._v(" "),s("p",[e._v("案例："),s("a",{attrs:{href:"https://zhijie-he.github.io/learning-blogs/",target:"_blank",rel:"noopener noreferrer"}},[e._v("https://zhijie-he.github.io/learning-blogs/"),s("OutboundLink")],1)]),e._v(" "),s("h2",{attrs:{id:"创建github库"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#创建github库"}},[e._v("#")]),e._v(" 创建github库")]),e._v(" "),s("p",[e._v("因为最终要将vuepress的网站部署到github pages上面，所以第一步首先创建GitHub库，如果不打算部署到github pages上面，可以跳过此步骤。")]),e._v(" "),s("h3",{attrs:{id:"创建public库"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#创建public库"}},[e._v("#")]),e._v(" 创建public库")]),e._v(" "),s("p",[e._v("对于部署到github pages上面的库要求必须是公开的。创建库完毕后，git到本地就可以进行本地编辑。这里以learning-blogs为例。")]),e._v(" "),s("h3",{attrs:{id:"文档结构"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#文档结构"}},[e._v("#")]),e._v(" 文档结构")]),e._v(" "),s("p",[e._v("git到刚才创建的库之后，")]),e._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[e._v("cd learning-blogs\n")])])]),s("p",[e._v("然后我们创建这样的目录结构")]),e._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[e._v(".\n├─ dev\n└─ docs\n└─ README.md\n└─ .gitignore\n")])])]),s("p",[e._v("dev目录作为我们的主要开发目录，也就是markdown文档以及vuepress代码的存储地方。docs文档这里与vuepress中的docs文档不同， 这个docs文档作为最终部署的文档，我们也称为docs.")]),e._v(" "),s("h2",{attrs:{id:"依赖环境"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#依赖环境"}},[e._v("#")]),e._v(" 依赖环境")]),e._v(" "),s("ul",[s("li",[e._v("Node.js")]),e._v(" "),s("li",[e._v("npm / yarn (推荐使用yarn)")])]),e._v(" "),s("h2",{attrs:{id:"创建项目"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#创建项目"}},[e._v("#")]),e._v(" 创建项目")]),e._v(" "),s("p",[e._v("以下命令如无特别指出，都在dev文件夹下运行：")]),e._v(" "),s("ol",[s("li",[e._v("初始化项目 "),s("code",[e._v("yarn init")])]),e._v(" "),s("li",[e._v("修改dev文件夹下 package.json文件中name的值为github repo的名称，即为"),s("code",[e._v("learning-blogs")])]),e._v(" "),s("li",[e._v("安装vuepress "),s("code",[e._v("yarn add -D vuepress")])]),e._v(" "),s("li",[e._v(".gitignore文件中忽略一些不必要的文件")])]),e._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[e._v("# VuePress files\ndocs/.vuepress/.temp/\ndocs/.vuepress/.cache/\ndocs/.vuepress/dist/\n\n# Node modules\nnode_modules/\n\n# MacOS Desktop Services Store\n.DS_Store\n\n# Log files\n*.log\n")])])]),s("p",[e._v("5.在package.json 中添加一些 scripts")]),e._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[e._v('{\n  "scripts": {\n    "docs:dev": "vuepress dev docs",\n    "docs:build": "vuepress build docs"\n  }\n}\n')])])]),s("p",[e._v("docs:dev和docs:build实际上是简化vuepress的指令。\n当我们在开发阶段，我们使用docs:dev模式，当我们开发完毕发版则用docs:build。")]),e._v(" "),s("p",[e._v("6.创建第一篇文档\n"),s("code",[e._v("echo '# Hello VuePress' > docs/README.md")])]),e._v(" "),s("ol",{attrs:{start:"7"}},[s("li",[e._v("运行\n"),s("code",[e._v("yarn docs:dev")])])]),e._v(" "),s("p",[e._v("VuePress 会在 "),s("a",{attrs:{href:"http://localhost:8080",target:"_blank",rel:"noopener noreferrer"}},[e._v("http://localhost:8080"),s("OutboundLink")],1),e._v(" 启动一个热重载的开发服务器。当你修改你的 Markdown 文件时，浏览器中的内容也会自动更新。")]),e._v(" "),s("p",[e._v("当看到这个地址下的Hello VuePress文字也就代表了vuepress初步完成。")]),e._v(" "),s("h2",{attrs:{id:"基础配置"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#基础配置"}},[e._v("#")]),e._v(" 基础配置")]),e._v(" "),s("p",[e._v("在docs文件夹下面创建一个.vuepress文件夹, 而文件夹下config.js文件则用来定义配置。")]),e._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[e._v(".\n├─ docs\n│  ├─ README.md\n│  └─ .vuepress\n│     └─ config.js\n└─ package.json\n")])])]),s("p",[e._v("一个 VuePress 网站必要的配置文件是 .vuepress/config.js，它应该导出一个 JavaScript 对象：")]),e._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[e._v("module.exports = {\n  title: 'Hello VuePress',\n  description: 'Just playing around'\n}\n")])])]),s("p",[e._v("对于上述的配置，如果你运行起 "),s("code",[e._v("yarn docs:dev")]),e._v("，你应该能看到一个页面，它包含一个页头，里面包含一个标题Hello VuePress和一个搜索框。VuePress 内置了基于 headers 的搜索 —— 它会自动为所有页面的标题、h2 和 h3 构建起一个简单的搜索索引。")]),e._v(" "),s("h3",{attrs:{id:"侧边栏sidebar"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#侧边栏sidebar"}},[e._v("#")]),e._v(" 侧边栏Sidebar")]),e._v(" "),s("p",[e._v("想要使侧边栏（Sidebar）生效，需要配置 themeConfig.sidebar，基本的配置，需要一个包含了多个链接的数组：")]),e._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[e._v("// .vuepress/config.js\nmodule.exports = {\n  themeConfig: {\n    sidebar: [\n      '/',\n      '/page-a',\n      ['/page-b', 'Explicit link text']\n    ]\n  }\n}\n")])])]),s("p",[e._v("可以省略 .md 拓展名，同时以 / 结尾的路径将会被视为 */README.md，这个链接的文字将会被自动获取到（无论你是声明为页面的第一个 header，还是明确地在 YAML front matter 中指定页面的标题）。如果你想要显示地指定链接的文字，使用一个格式为 [link, text] 的数组。")]),e._v(" "),s("p",[s("code",[e._v("注意：侧边栏的文字如无制定，默认是由指定页面的H1大标题决定")])]),e._v(" "),s("h2",{attrs:{id:"部署到github-pages"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#部署到github-pages"}},[e._v("#")]),e._v(" 部署到github pages")]),e._v(" "),s("p",[e._v("项目开发完毕后，使用"),s("code",[e._v("yarn docs:build")]),e._v("构建项目，会在"),s("code",[e._v("dev/docs/.vuepress/dist")]),e._v("文件夹下构建最终的页面，然后我们将dist文件加下的所有文件去复制到与最初dev同级目录下的docs文件。然后将文件上传到github库。")]),e._v(" "),s("h3",{attrs:{id:"设置github-pages"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#设置github-pages"}},[e._v("#")]),e._v(" 设置github pages")]),e._v(" "),s("ol",[s("li",[e._v("打开github对应的库，并点击settings")]),e._v(" "),s("li",[e._v("在侧边栏找到Code and automation栏目，点击Pages")]),e._v(" "),s("li",[e._v("设置Branch信息，选择main brance并且设置文件目录为与dev文件夹同级目录的docs目录")]),e._v(" "),s("li",[e._v("保存之后，会给出网站链接，等待刷新即部署成功。")]),e._v(" "),s("li",[e._v("再次更新vuepress内容，只需要构建完成后，复制dist文件内容到docs文件，然后push到github，即可自动部署。")])]),e._v(" "),s("h2",{attrs:{id:"faq"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#faq"}},[e._v("#")]),e._v(" FAQ")]),e._v(" "),s("ol",[s("li",[e._v("使用 yarn docs:build 运行 VuePress 时遇到的 Error: error:0308010C:digital envelope routines::unsupported 错误")])]),e._v(" "),s("blockquote",[s("p",[e._v("出现这个原因主要是因为node的版本太高而yarn的版本太低，目前我使用的node版本的是v21.6.1 yarn的版本为：1.22.22。 解决办法为将 NODE_OPTIONS 环境变量设置为 --openssl-legacy-provider 是解决这个问题的常见方法。这会告诉 Node.js 使用旧版的加密提供程序，这在新版本中可能默认不支持。")])]),e._v(" "),s("ul",[s("li",[s("p",[e._v("对于 macOS/Linux：\n打开您的终端并运行以下命令： "),s("code",[e._v("export NODE_OPTIONS=--openssl-legacy-provider")])])]),e._v(" "),s("li",[s("p",[e._v("对于 Windows：\n打开命令提示符并执行："),s("code",[e._v("set NODE_OPTIONS=--openssl-legacy-provider")])])])])])}),[],!1,null,null,null);s.default=r.exports}}]);