module.exports={
    title:'hello vuepress',
    description:'just play around',
    base:'/learning-blogs/',
    themeConfig:{
        sidebar:[
            '/',
            '/category/',
            '/about/',
            {
                title:'category_nest',
                path:'/category/',
                children:[
                    "/category/test1.md",
                    "/category/test2.md"
                ]
            }
        ]
    }

}