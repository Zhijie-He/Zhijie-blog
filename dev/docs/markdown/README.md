# Markdown 教程

- [Markdowm tutorial](https://commonmark.org/help/)
## 强调
使用*或者_去创建加粗或者斜体。

单组*或者_是实现斜体，双组则是加粗。

```
在*或者_前面加上\就可以展示*原来的值。
*italics*  \*italics\*
```

*italics*  \*italics\*


## 换行
如果要实现换行，可以通过加上\或者两个空行。

```
The sky above the port was the color of television, tuned to a dead channel.

It was a bright cold day in April, and the clocks were striking thirteen.
```

```
I have eaten\
the plums\
that were in\
the icebox
```

## 标题
通过#开头构建标题。

## 引用
通过使用>加上一个空格来形成引用。并且引用可以嵌套。
```
The quote

> Somewhere, something incredible is waiting to be known

has been ascribed to Carl Sagan.
```

The quote

> Somewhere, something incredible is waiting to be known

has been ascribed to Carl Sagan.

```
My favorite Miss Manners quotes:

> Allowing an unimportant mistake to pass without comment is a wonderful social grace.
>
> Ideological differences are no excuse for rudeness.
```
My favorite Miss Manners quotes:

> Allowing an unimportant mistake to pass without comment is a wonderful social grace.
>
> Ideological differences are no excuse for rudeness.


## 列表
无序列表使用：*、+、-作为列表标记

有序列表使用数字后面跟上.或者)

```
+ Flour
+ Cheese
+ Tomatoes
```

```
Four steps to better sleep:
1. Stick to a sleep schedule
2. Create a bedtime ritual
3. Get comfortable
4. Manage stress
```

1986. What a great season. Arguably the finest season in the history of the franchise.

如果这种类型的不想要列表，则需要在.前面加上\.



1986\. What a great season. Arguably the finest season in the history of the franchise.

## 链接
URL链接需要被包括在<>标签里面才能成为链接。
```
You can do anything at <https://html5zombo.com>
```
You can do anything at <https://html5zombo.com>

内嵌链接的方式
```
The [University of Rwanda](http://www.ur.ac.rw) was formed in 2013 through the merger of Rwanda’s seven public institutions of higher education.
```
也可以使用类似reference的格式创建链接

```
[Hurricane][1] Erika was the strongest and longest-lasting tropical cyclone in the 1997 Atlantic [hurricane][1] season.

[1]:https://w.wiki/qYn
```
[Hurricane][1] Erika was the strongest and longest-lasting tropical cyclone in the 1997 Atlantic [hurricane][1] season.

[1]:https://w.wiki/qYn

## 图片
和链接很相似，唯一的区别是图片前面有！

```
![](https://commonmark.org/help/images/favicon.png)
```
![](https://commonmark.org/help/images/favicon.png)


可以给图片加上alt 文字
```
![Logo][1]

[1]: https://commonmark.org/help/images/favicon.png "Creative Commons licensed"
```
![Logo][1]

[1]: https://commonmark.org/help/images/favicon.png "Creative Commons licensed"

## 代码行
使用``可以创建单行的代码。
使用四个缩进或者连续的三个```可以构建块状代码。
注意在```代码块后面加上具体的语言，即可展示具体语言的代码块，比如python:

```python
x=1
y=1
print(x+y)
```

```
When `x = 3`, that means `x + 2 = 5`
```
When `x = 3`, that means `x + 2 = 5`

```
Who ate the most donuts this week?

    Jeff  15
    Sam   11
    Robin  6
```

Who ate the most donuts this week?

    Jeff  15
    Sam   11
    Robin  6

A loop in JavaScript:
```
var i;
for (i=0; i<5; i++) {
  console.log(i);
}
```
What numbers will this print?

## 嵌套列表

```
* Fruit
  * Apple
  * Orange
  * Banana
* Dairy
  * Milk
  * Cheese
```
* Fruit
  * Apple
  * Orange
  * Banana
* Dairy
  * Milk
  * Cheese

```
+ World Cup 2014
  1. Germany
  2. Argentina
  3. Netherlands
+ Rugby World Cup 2015
  1. New Zealand
  2. Australia
  3. South Africa
```
+ World Cup 2014
  1. Germany
  2. Argentina
  3. Netherlands
+ Rugby World Cup 2015
  1. New Zealand
  2. Australia
  3. South Africa

```
1. Ingredients

    - spaghetti
    - marinara sauce
    - salt

2. Cooking

   Bring water to boil, add a pinch of salt and spaghetti. Cook until pasta is **tender**.

3. Serve

   Drain the pasta on a plate. Add heated sauce. 

   > No man is lonely eating spaghetti; it requires so much attention.

   Bon appetit!
```
1. Ingredients

    - spaghetti
    - marinara sauce
    - salt

2. Cooking

   Bring water to boil, add a pinch of salt and spaghetti. Cook until pasta is **tender**.

3. Serve

   Drain the pasta on a plate. Add heated sauce. 

   > No man is lonely eating spaghetti; it requires so much attention.

   Bon appetit!