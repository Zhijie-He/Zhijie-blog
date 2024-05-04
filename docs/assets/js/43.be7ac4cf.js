(window.webpackJsonp=window.webpackJsonp||[]).push([[43],{333:function(t,a,r){"use strict";r.r(a);var e=r(14),n=Object(e.a)({},(function(){var t=this,a=t._self._c;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"安装"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#安装"}},[t._v("#")]),t._v(" 安装")]),t._v(" "),a("p",[t._v("这里主要介绍PyTorch的基础教学以及如何使用Apple M1芯片进行Machine learning的训练。")]),t._v(" "),a("blockquote",[a("p",[t._v("Until now, PyTorch training on Mac only leveraged the CPU, but with the upcoming PyTorch v1.12 release, developers and researchers can take advantage of Apple silicon GPUs for significantly faster model training.\nAccelerated GPU training is enabled using Apple’s Metal Performance Shaders (MPS) as a backend for PyTorch.")])]),t._v(" "),a("h2",{attrs:{id:"资源"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#资源"}},[t._v("#")]),t._v(" 资源")]),t._v(" "),a("ul",[a("li",[a("a",{attrs:{href:"https://developer.apple.com/metal/pytorch/",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://developer.apple.com/metal/pytorch/"),a("OutboundLink")],1)]),t._v(" "),a("li",[a("a",{attrs:{href:"https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/"),a("OutboundLink")],1)]),t._v(" "),a("li",[a("a",{attrs:{href:"https://medium.com/@manyi.yim/pytorch-on-mac-m1-gpu-installation-and-performance-698442a4af1e",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://medium.com/@manyi.yim/pytorch-on-mac-m1-gpu-installation-and-performance-698442a4af1e"),a("OutboundLink")],1)]),t._v(" "),a("li",[t._v("torch算术运算 "),a("a",{attrs:{href:"https://pytorch.org/docs/stable/torch.html",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://pytorch.org/docs/stable/torch.html"),a("OutboundLink")],1)]),t._v(" "),a("li",[t._v("UvA Deep Learning Tutorials "),a("a",{attrs:{href:"https://uvadlc-notebooks.readthedocs.io/en/latest/index.html",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://uvadlc-notebooks.readthedocs.io/en/latest/index.html"),a("OutboundLink")],1)]),t._v(" "),a("li",[t._v("official tutorials "),a("a",{attrs:{href:"https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html"),a("OutboundLink")],1)])]),t._v(" "),a("h2",{attrs:{id:"macos-安装pytorch"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#macos-安装pytorch"}},[t._v("#")]),t._v(" MacOS 安装PyTorch")]),t._v(" "),a("blockquote",[a("p",[t._v("To get started, just install the latest Preview (Nightly) build on your Apple silicon Mac running macOS 12.3 or later with a native version (arm64) of Python.")])]),t._v(" "),a("p",[t._v("这里使用conda进行安装。对于conda的使用与安装可以查看该教程："),a("a",{attrs:{href:"/coding-tips/conda"}},[t._v("Conda")])]),t._v(" "),a("ul",[a("li",[t._v("安装PyTorch")])]),t._v(" "),a("p",[t._v("到"),a("a",{attrs:{href:"https://pytorch.org/get-started/locally/",target:"_blank",rel:"noopener noreferrer"}},[t._v("PyTorch安装官网"),a("OutboundLink")],1),t._v("，去查看对应的安装命令, 这里以Mac M1为例：")]),t._v(" "),a("div",{staticClass:"language-python extra-class"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[t._v("conda install pytorch"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("nightly"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("pytorch torchvision torchaudio "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c pytorch"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("nightly\n")])])]),a("h3",{attrs:{id:"验证"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#验证"}},[t._v("#")]),t._v(" 验证")]),t._v(" "),a("div",{staticClass:"language-python extra-class"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">>")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" torch\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">>")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("print")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("backends"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mps"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("is_available"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("True")]),t._v("\n")])])])])}),[],!1,null,null,null);a.default=n.exports}}]);