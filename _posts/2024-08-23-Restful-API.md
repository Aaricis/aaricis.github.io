---
title: Restful API
date: 2024-08-23 16:40:00 +/-8
categories: [Product]
tags: [restful api]     # TAG names should always be lowercase
---

#  Restful风格

https://webkul.com/blog/cs-cart-rest-api/

## REST 成熟度模型

参考： https://martinfowler.com/articles/richardsonMaturityModel.html

**URI命名围绕资源进行设计**

 

**用http方法表示动作：**

GET,POST,PUT,DELETE,PATCH

 

**资源名称简短有意义(但不是简写)，是具体的，不是抽象的**

- ✔️/users 、 /role-groups 
- ❌/cso  /cce 
- ✔️/tasks 
- ❌/resources ---过于抽象 

**使用名词的复数形式，而不是动词或动宾短语，建议全小写**

- ✔️POST: /users 
- ✔️DELETE :/teams/{id} 
- ❌POST:/createUser 
- ❌DELETE :/deleteTeam 
- ❌GET:/getUserById 
- ❌GET:/getUserByW3Account 

**复合名词，用-分割，注意不是_**

- ✔️GET: /master-data/accounts 
- ❌GET:/master_data/accounts 

**扁平化胜过嵌套（取决于聚合根）**

- ❌/users/123/roles/456/permissions/100 
- ✔️/permissions?user-id=123&role-id=456 

**注意http状态码和响应体的错误码的区别**

- 注意401和403状态码的区别：401-未认证和403-未授权 
- 响应体的错误信息不要暴露调用堆栈信息 

**不要返回纯文本型，返回的结果应该是json格式对象**

- code：0或自定义的错误码 
- message：ok/error信息 
- data:{...}：有效的业务对象 

**自定义动作放到资源结尾:分割，例如**

- ✔️ 导出：POST /meta-data/permissions:export  --生成导出文件 
- ✔️ 移动：PATCH /documents:move  ---等同于更新父节点 
- ❌ 移动：POST /moveDocumentToPath 
- ✔️ 取消：POST /tasks:cancel  ---取消一个未完成的任务，会增加一个条取消记录 
- ❌取消：DELETE /cancelTask 

**不要在尾部增加斜杠：**

- ❌/users/ 
- ✔️/users

## 附录：

**一、Restful VS RPC**

| 比较项 | REST                                                         | RPC                                                          |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 描述   | 分布式多媒体系统设计的一组架构约束的集合                     | 一个计算机通信协议，允许从一台计算访问另外一台计算机上的程序 |
| 特征   | 以资源为中心<br/>统一的行为<br/>标准的媒体类型<br/>大量的资源<br/>通用知识 | 自定义行为<br/> 自定义媒体类型<br/> 少量的对象<br/>个性化知识 |

**二、HTTP方法**

| 方法    | 描述                                                         | SQL映射 | 增删改查CRUD |
| ------- | ------------------------------------------------------------ | ------- | ------------ |
| POST    | 根据客户端提供的数据创建一个新的资源                         | Insert  | Create       |
| GET     | 返回资源当前展现                                             | Select  | Retrieve     |
| PUT     | 根据客户端提供的数据替换指定资源                             | Update  | Update       |
| DELETE  | 删除某个资源                                                 | Delete  | Delete       |
| HEAD    | 返回资源的元数据，比如etag,last-modified之类的信息，支持GET请求的资源，一般也会有HEAD方法 | Select  | Retrieve     |
| PATCH   | 部分更新资源                                                 | Update  | Update       |
| OPTIONS | 获取当前资源信息，比如当前资源支持哪些方法的信息。           | Select  | Retrieve     |

**三、参考样例**

| 资源                | POST                | GET                   | PUT                               | DELETE                |
| ------------------- | ------------------- | --------------------- | --------------------------------- | --------------------- |
| /customers          | 创建新客户          | 检索所有客户          | 批量更新客户                      | 删除所有客户          |
| /customers/1        | 错误                | 检索客户 1 的详细信息 | 如果客户 1 存在，则更新其详细信息 | 删除客户 1            |
| /customers/1/orders | 创建客户 1 的新订单 | 检索客户 1 的所有订单 | 批量更新客户 1 的订单             | 删除客户 1 的所有订单 |

**四、REST API 设计思维导图**

https://architect.pub/book/export/html/1641