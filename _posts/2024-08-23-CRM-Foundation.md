---
title: CRM Foundation
date: 2024-08-23 14:40:00 +/-8
categories: [Product, CRM]
tags: [crm]     # TAG names should always be lowercase
---

# **概述**

为实现“统一技术栈、安全、高效上云&用云”，为应用现代化使能，Meta系列产品统一构建SaaS Foundation。



CRM 公共业务服务 作为 Meta SaaS Foundation 针对 CRM产品的补充，针对CRM的共性需求构建出的一组公共数据、公共能力及增值业务的服务。

CRM 公共业务服务 是Meta CRM所有Cloud的开发时的公共底座与运行时依赖的公共服务。

如下图所示：有颜色的块为CRM 公共业务服务



**客户端**

SDK组件，被各个Cloud的服务所依赖，通过SDK可以以本地方法的形式访问Foundation的远程服务。

Web页面，被CRM 门户集成，跨多个Cloud的一组公共的后台管理界面，如个人设置，管理后台。

Web卡片，被各个Cloud的页面所引用的公共卡片组件，Foundation增值服务(如任务、消息等）主要以卡片形式被集成在各Cloud中。

 

**服务端**

独立部署的一组公共服务，主要分类两类：

查询服务：面向机机接口，针对各Cloud的高频访问接口，提供高性能的查询服务。

管理服务：面向人机接口，主要针对租户管理员、开发/实施人员的后台管理、配置服务。

|                                            | SDK接口 | 查询服务 | 管理服务 | 管理页面/卡片 |
| ------------------------------------------ | ------- | -------- | -------- | ------------- |
| 公共数据(主数据/基础数据/元数据)           | ⭐⭐⭐⭐    | ⭐⭐⭐      | ⭐⭐⭐      | ⭐⭐⭐           |
| 公共能力(权限、审计、合规、设置、用户租户) | ⭐⭐⭐     | ⭐⭐       | ⭐⭐⭐⭐     | ⭐⭐⭐⭐          |
| 增值业务(配置、协同、自动化)               | ⭐       | ⭐        | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐         |

# 调用关系



#  CRM SDK

CRM SDK 是CRM公共技术组件，旨在规范CRM产品代码开发，提升业务开发效率。

## 模块说明

### Dependencies

管理依赖的二方件、三方件及可信版本，作为所有的CRM 业务工程的公共父POM

### Base

提供包括异常、日志、工具、安全类、消息总线、上下文等基本的能力

### Base Extensions

针对基础组件的扩展，包括正确性检验规则、唯一性校验规则、敏感词过滤等

### DDD

基于领域驱动设计的基础框架，包括命令、应用服务、领域事件、领域模型、资源库等基类

### Cloud Connectors

适配云专属服务的链接器，包括：登录、文档、敏感词扫描、配置中心等

### Foundation Clients

Foundation 服务的客户端，包括 ：

- 公共数据（Command Data 简写为 cmd） 
  - Reference Data: 基础数据 
  - Master Data: 主数据 
  - Meta Data: 元数据 
- 公共能力（Command Ability 简写为 cma） 
  - Authentication & Authorization：认证授权 
  - Audit Trail ：审计追溯 
  - Privacy Compliance：隐私合规 
- 增值业务（Extended Business 简写为 exb） 
  - Configuration：配置 
  - Collaboration：协同 
  - Automation：自动化 

### Autoconfigure

SPI的默认Provider的自动配置

### Starters

组合后的启动器，提供给CRM各产品服务依赖，实现开箱即用
