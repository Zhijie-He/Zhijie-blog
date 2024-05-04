module.exports = {
    title: "Zhijie He's Blog",
    description: "Zhijie He's learning blogs in computer science",

    base: '/learning-blogs/',
    // theme: 'reco',
    themeConfig: {
        theme: 'reco',
        nav: [
            // { text: 'My Website', link: 'https://zhijie-he.github.io/' },
            // { text: '创业之路', link: '/start-up/' },
            { text: 'Github', link: 'https://github.com/Zhijie-He/learning-blogs' }
        ],

        sidebar: [
            {
                title: '欢迎学习',
                path: '/',
                collapsable: false, // 不折叠
                children: [
                    { title: "学前必读", path: "/" }
                ]
            },
            {
                title: "VuePress",
                path: '/vuepress',
                // collapsable: false, // 不折叠
                children: [
                    "/vuepress/vuepress-starter",
                    "/vuepress/vuepress-basics",
                    "/vuepress/vuepress-pro"
                ],

            },

            '/markdown/',
            {
                title: "微信小程序",
                path: '/miniprograms',
                // collapsable: false, // 不折叠
                children: [
                    "/vuepress/vuepress-starter",

                ],

            },
            {
                title: "博士",
                path: '/phd',
                // collapsable: false, // 不折叠
                children: [
                    "/phd/prepare",

                ],

            },
            {
                title: "编程技巧",
                path: '/coding-tips',
                // collapsable: false, // 不折叠
                children: [
                    "/coding-tips/git",
                    "/coding-tips/conda",
                    "/coding-tips/python",
                    "/coding-tips/mac"

                ],

            },
            {
                title: "PyTorch",
                path: '/pytorch',
                // collapsable: false, // 不折叠
                children: [
                    '/pytorch/',
                    "/pytorch/tutorial",
                    "/pytorch/hyperparameter_tuning",
                    "/pytorch/optimizer",
                    "/pytorch/transforms",
                    "/pytorch/applications",

                ],
            },
            {
                title: "PyTorch Timm",
                path: '/pytorch-timm',
                // collapsable: false, // 不折叠
                children: [
                    '/pytorch-timm/',
                    '/pytorch-timm/basics',
                ],

            },
            {
                title: "PyTorch Lightning",
                path: '/pytorch-lightning',
                // collapsable: false, // 不折叠
                children: [
                    '/pytorch-lightning/',
                    '/pytorch-lightning/basics',
                    '/pytorch-lightning/intermediate',
                    '/pytorch-lightning/applications',
                ],

            },
            {
                title: "深度学习教程",
                path: '/UvA-DL-notebooks',
                // collapsable: false, // 不折叠
                children: [
                    '/UvA-DL-notebooks/',
                ],

            },
            
            
        ]
    }

}