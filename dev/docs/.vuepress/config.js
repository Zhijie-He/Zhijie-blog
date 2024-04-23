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
                    "/vuepress/vuepress-starter"
                ],

            },
            
           '/markdown/',
        ]
    }

}