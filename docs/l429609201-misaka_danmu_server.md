# 御坂网络弹幕服务

[![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github)](https://github.com/l429609201/misaka_danmu_server)
![GitHub License](https://img.shields.io/github/license/l429609201/misaka_danmu_server)
![Docker Pulls](https://img.shields.io/docker/pulls/l429609201/misaka_danmu_server)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/l429609201/misaka_danmu_server?color=blue&label=download&sort=semver)](https://github.com/l429609201/misaka_danmu_server/releases/latest)
[![telegram](https://img.shields.io/static/v1?label=telegram&message=misaka_danmu_server&color=blue)](https://t.me/misaka_danmu_server)



一个功能强大的自托管弹幕（Danmaku）聚合与管理服务，兼容 [dandanplay](https://api.dandanplay.net/swagger/index.html) API 规范。

本项目旨在通过刮削主流视频网站的弹幕，为您自己的媒体库提供一个统一、私有的弹幕API。它自带一个现代化的Web界面，方便您管理弹幕库、搜索源、API令牌和系统设置。



> [!IMPORTANT]
> **按需使用，请勿滥用**
> 本项目旨在作为个人媒体库的弹幕补充工具。所有弹幕数据均实时从第三方公开API或网站获取。请合理使用，避免对源站造成不必要的负担。过度频繁的请求可能会导致您的IP被目标网站屏蔽。

> [!NOTE]
> **网络与地区限制**
> 推荐使用大陆IP使用本项目，如果您在海外地区部署或使用此项目，可能会因网络或地区版权限制导致无法访问这些视频源。建议海外用户配置代理或确保网络环境可以访问国内网站。

## ✨ 核心功能

- **智能匹配**: 通过文件名或元数据（TMDB, TVDB等）智能匹配您的影视文件，提供准确的弹幕。
- **Web管理界面**: 提供一个直观的Web UI，用于：
  - 搜索和手动导入弹幕。
  - 管理已收录的媒体库、数据源和分集。
  - 创建和管理供第三方客户端（如 yamby, hills, 小幻影视）使用的API令牌。
  - 配置搜索源的优先级和启用状态。
  - 查看后台任务进度和系统日志。
- **元数据整合**: 支持与 TMDB, TVDB, Bangumi, Douban, IMDb 集成，丰富您的媒体信息。
- **自动化**: 支持通过 Webhook 接收来自 Sonarr, Radarr, Emby 等服务的通知，实现全自动化的弹幕导入。
- **灵活部署**: 提供 Docker 镜像和 Docker Compose 文件，方便快速部署。

## 其他

### 免责声明

在使用本项目前，请您务必仔细阅读并理解本声明。一旦您选择使用，即表示您已充分理解并同意以下所有条款。

#### 1. 项目性质

- **技术中立性**: 本项目是一个开源的、自托管的技术工具，旨在通过自动化程序从公开的第三方视频网站、公开的API中获取弹幕评论数据。
- **功能范围**: 本工具仅提供弹幕数据的聚合、存储和API访问功能，供用户在个人合法拥有的媒体上匹配和加载，以提升观影体验。
- **非内容提供方**: 本项目不生产、不修改、不存储、不分发任何视频内容本身，所有弹幕内容均来源于第三方平台的公开分享。

#### 2. 用户责任

- **遵守服务条款**: 您理解并同意，抓取第三方网站数据的行为可能违反其服务条款（ToS）。您承诺将自行承担因使用本工具而可能引发的任何风险，包括但不限于来自源网站的警告、账号限制或法律追究。
- **内容风险自负**: 所有弹幕均为第三方平台用户公开发布，其内容（可能包含不当言论、剧透、广告等）的合法性、真实性及安全性由发布者独立负责。您需自行判断并承担查看这些内容可能带来的所有风险。
- **合法合规使用**: 您承诺仅将本工具用于个人学习、研究或非商业用途，并遵守您所在国家/地区的相关法律法规，不得将本工具及获取的数据用于任何非法或侵权活动。

#### 3. 开发者免责

- **内容无关性**: 开发者仅提供技术实现，不参与任何弹幕内容的创作、审核、编辑或推荐，亦无法保证弹幕的准确性、完整性、实时性或质量。
- **服务不保证**: 由于本项目依赖第三方网站的接口和数据结构，开发者无法保证服务的永久可用性。任何因源网站API变更、反爬虫策略升级、网络环境变化或不可抗力导致的服务中断或功能失效，开发者不承担任何责任。
- **免责范围**: 在法律允许的最大范围内，开发者不对以下情况负责：
    - 您因违反第三方网站服务条款而导致的任何损失或法律后果。
    - 您因接触或使用弹幕内容而产生的任何心理或生理不适。
    - 因使用本工具导致的任何直接、间接、偶然或必然的设备损坏或数据丢失。
- **权利保留**: 开发者保留随时修改、更新或终止本项目的权利，恕不另行通知。

---

### 推广须知

- 请不要在 ***B站*** 或中国大陆社交平台发布视频或文章宣传本项目

## 🚀 快速开始 (使用 Docker Compose)


推荐使用 Docker 和 Docker Compose 进行一键部署。

### 步骤 1: 准备 `docker-compose.yaml`

1.  在一个合适的目录（例如 `~/danmuku`）下，创建 `docker-compose.yaml` 文件和所需的文件夹 `config，db-data`。


    ```bash
    mkdir -p ~/danmuku
    cd ~/danmuku
    mkdir db-data,config                 
    touch docker-compose.yaml
    ```

2.  根据您选择的数据库，将以下内容之一复制到 `docker-compose.yaml` 文件中。

#### 方案 A: 使用 MySQL (推荐)


```yaml
version: "3.8"
services:
  mysql:
    image: mysql:8.1.0-oracle
    container_name: danmu-mysql
    restart: unless-stopped
    environment:
      # !!! 重要：请务必替换为您的强密码 !!!
      MYSQL_ROOT_PASSWORD: "your_strong_root_password"                  #数据库root密码
      MYSQL_DATABASE: "danmuapi"                                        #数据库名称
      MYSQL_USER: "danmuapi"                                            #数据库用户名
      MYSQL_PASSWORD: "your_strong_user_password"                       #数据库密码
      TZ: "Asia/Shanghai"
    volumes:
      - ./db-data:/var/lib/mysql
    command:
      - '--character-set-server=utf8mb4'
      - '--collation-server=utf8mb4_unicode_ci'
      - '--expire_logs_days=3' # 自动清理超过3天的binlog日志
      - '--binlog_expire_logs_seconds=259200' # 兼容MariaDB的等效设置 (3天)
    healthcheck:
      # 使用mysqladmin ping命令进行健康检查，通过环境变量引用密码
      test: ["CMD-SHELL", "mysqladmin ping -u$$MYSQL_USER -p$$MYSQL_PASSWORD"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 30s

    networks:
      - misaka-net

  danmu-app:
    image: l429609201/misaka_danmu_server:latest
    container_name: misaka-danmu-server
    restart: unless-stopped
    depends_on:
      mysql:
        condition: service_healthy
    environment:
      # 设置运行容器的用户和组ID，以匹配您宿主机的用户，避免挂载卷的权限问题。
      - PUID=1000
      - PGID=1000
      - UMASK=0022
      - TZ=Asia/Shanghai
      # --- 数据库连接配置 ---
      - DANMUAPI_DATABASE__TYPE=mysql                         # 数据库类型
      - DANMUAPI_DATABASE__HOST=mysql                         # 使用服务名
      - DANMUAPI_DATABASE__PORT=3306                          # 端口号
      - DANMUAPI_DATABASE__NAME=danmuapi                      # 数据库名称
      # !!! 重要：请使用上面mysql容器相同的用户名和密码 !!!
      - DANMUAPI_DATABASE__USER=danmuapi                      #数据库用户名
      - DANMUAPI_DATABASE__PASSWORD=your_strong_user_password #数据库密码
      # --- 初始管理员配置 ---
      - DANMUAPI_ADMIN__INITIAL_USER=admin
    volumes:
      - ./config:/app/config
    ports:
      - "7768:7768"
    networks:
      - misaka-net

networks:
  misaka-net:
    driver: bridge
```

#### 方案 B: 使用 PostgreSQL (可选)

```yaml
version: "3.8"
services:
  postgres:
    image: postgres:16
    container_name: danmu-postgres
    restart: unless-stopped
    environment:
      # !!! 重要：请务必替换为您的强密码 !!!
      POSTGRES_PASSWORD: "your_strong_postgres_password"               #数据库密码
      POSTGRES_USER: "danmuapi"                                        #数据库用户名
      POSTGRES_DB: "danmuapi"                                          #数据库名称
      TZ: "Asia/Shanghai"
    volumes:
      - ./db-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U danmuapi -d danmuapi"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 30s
    networks:
      - misaka-net

  danmu-app:
    image: l429609201/misaka_danmu_server:latest
    container_name: misaka-danmu-server
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      # 设置运行容器的用户和组ID，以匹配您宿主机的用户，避免挂载卷的权限问题。
      - PUID=1000
      - PGID=1000
      - UMASK=0022
      - TZ=Asia/Shanghai
      # --- 数据库连接配置 ---
      - DANMUAPI_DATABASE__TYPE=postgresql                              # 数据库类型
      - DANMUAPI_DATABASE__HOST=postgres                                # 使用服务名
      - DANMUAPI_DATABASE__PORT=5432                                    # 数据库端口
      - DANMUAPI_DATABASE__NAME=danmuapi                                # 数据库名称
      # !!! 重要：请使用上面postgres容器相同的用户名和密码 !!!
      - DANMUAPI_DATABASE__USER=danmuapi                                # 数据库用户名    
      - DANMUAPI_DATABASE__PASSWORD=your_strong_postgres_password       # 数据库密码
      # --- 初始管理员配置 ---
      - DANMUAPI_ADMIN__INITIAL_USER=admin
    volumes:
      - ./config:/app/config
    ports:
      - "7768:7768"

    networks:
      - misaka-net

networks:
  misaka-net:
    driver: bridge

```

### 步骤 2: 修改配置并启动

1.  **重要**: 打开您刚刚创建的 `docker-compose.yaml` 文件，将所有 `your_strong_..._password` 替换为您自己的安全密码。
    -   对于MySQL，您需要修改 `MYSQL_ROOT_PASSWORD`, `MYSQL_PASSWORD` (两处) 和 `healthcheck` 中的密码。
    -   对于PostgreSQL，您需要修改 `POSTGRES_PASSWORD` 和 `DANMUAPI_DATABASE__PASSWORD`。
2.  在 `docker-compose.yaml` 所在目录运行命令启动应用：
    ```bash
    docker-compose up -d

    ```

### 步骤 3: 访问和配置

- **访问Web UI**: 打开浏览器，访问 `http://<您的服务器IP>:7768`。
- **初始登录**:
  - 用户名: `admin` (或您在环境变量中设置的值)。
  - 密码: 首次启动时会在容器的日志中生成一个随机密码。请使用 `docker logs misaka-danmu-server` 查看。
- **开始使用**: 登录后，请先在 "设置" -> "账户安全" 中修改您的密码，然后在 "搜索源" 和 "设置" 页面中配置您的API密钥。

## 客户端配置

### 1. 获取弹幕 Token

- 在 Web UI 的 "弹幕Token" 页面，点击 "添加Token" 来创建一个新的访问令牌。
- 创建后，您会得到一串随机字符，这就是您的弹幕 Token。
- 可通过配置自定义域名之后直接点击复制，会帮你拼接好相关的链接

### 2. 配置弹幕接口

在您的播放器（如 Yamby, Hills, 小幻影视等）的自定义弹幕接口设置中，填入以下格式的地址：

`http://<服务器IP>:<端口>/api/v1/<你的Token>`

- `<服务器IP>`: 部署本服务的主机 IP 地址。
- `<端口>`: 部署本服务时设置的端口（默认为 `7768`）。
- `<你的Token>`: 您在上一步中创建的 Token 字符串。

**示例:**

假设您的服务部署在 `192.168.1.100`，端口为 `7768`，创建的 Token 是 `Q2KHYcveM0SaRKvxomQm`。


- **对于 Yamby （版本要大于1.5.9.11） / Hills （版本要大于1.4.2）:**

  在自定义弹幕接口中填写：
  `http://192.168.1.100:7768/api/v1/Q2KHYcveM0SaRKvxomQm`
- **对于 小幻影视:**
  小幻影视您可以添加含有 `/api/v2` 的路径，可以直接填写复制得到的url：
  `http://192.168.1.100:7768/api/v1/Q2KHYcveM0SaRKvxomQm/api/v2   #可加可不加/api/v2 ` 
  
> **兼容性说明**: 本服务已对路由进行特殊处理，无论您使用 `.../api/v1/<Token>` 还是 `.../api/v1/<Token>/api/v2` 格式，服务都能正确响应，以最大程度兼容不同客户端。

## Webhook 配置

本服务支持通过 Webhook 接收来自 Emby 等媒体服务器的通知，实现新媒体入库后的弹幕自动搜索和导入。

### 1. 获取 Webhook URL

1. 在 Web UI 的 "设置" -> "Webhook" 页面，您会看到一个为您生成的唯一的 **API Key**。
2. 根据您要集成的服务，复制对应的 Webhook URL。URL 的通用格式为：
   `http://<服务器IP>:<端口>/api/webhook/{服务名}?api_key=<你的API_Key>`

   - `<服务器IP>`: 部署本服务的主机 IP 地址。
   - `<端口>`: 部署本服务时设置的端口（默认为 `7768`）。
   - `{服务名}`: webhook界面中下方已加载的服务名称，例如 `emby`。
   - `<你的API_Key>`: 您在 Webhook 设置页面获取的密钥。
3. 现在已经增加拼接URL后的复制按钮

### 2. 配置媒体服务器

- **对于Emby**

  1. 登录您的 Emby 服务器管理后台。
  2. 导航到 **通知** (Notifications)。
  3. 点击 **添加通知** (Add Notification)，选择 **Webhook** 类型。
  4. 在 **Webhook URL** 字段中，填入您的 Emby Webhook URL，例如：
     ```
     http://192.168.1.100:7768/api/webhook/emby?api_key=your_webhook_api_key_here
     ```
  5. **关键步骤**: 在 **事件** (Events) 部分，请务必**只勾选**以下事件：
     - **项目已添加 (Item Added)**: 这是新媒体入库的事件，其对应的事件名为 `新媒体添加`。
  6. 确保 **发送内容类型** (Content type) 设置为 `application/json`。
  7. 保存设置。
- **对于Jellyfin**

  1. 登录您的 Jellyfin 服务器管理后台。
  2. 导航到 **我的插件**，找到 **Webhook** 插件，如果没有找到，请先安装插件，并重启服务器。
  3. 点击 **Webhook** 插件，进入配置页面。
  4. 在 **Server Url** 中输入jellyfin 访问地址（可选）。
  5. 点击 **Add Generic Destination**。
  6. 输入 **Webhook Name**
  7. 在 **Webhook URL** 字段中，填入您的 Jellyfin Webhook URL，例如：
     ```
     http://192.168.1.100:7768/api/webhook/jellyfin?api_key=your_webhook_api_key_here
     ```
  8. **关键步骤**: 在 **Notification Type** 部分，请务必**只勾选**以下事件：
     - **Item Added**: 这是新媒体入库的事件，其对应的事件名为 `新媒体添加`。
  9. **关键步骤**: 一定要勾选 **Send All Properties (ignores template)** 选项。
  10. 保存设置。

现在，当有新的电影或剧集添加到您的 Emby/Jellyfin 媒体库时，本服务将自动收到通知，并创建一个后台任务来为其搜索和导入弹幕。

## 🤖 Telegram Bot 集成

[balge](https://github.com/balge) 开发了一个功能强大的 Telegram Bot，可以帮助您通过聊天界面管理您的弹幕服务器。（弹幕库版本要大于v2.0.4才可以使用）

**项目地址**: [misaka-danmuku-bot](https://github.com/balge/misaka-danmuku-bot)

通过此机器人，您可以：
- 搜索和导入新的影视作品。
- 管理媒体库、数据源和分集。
- 查看和管理后台任务。


## 常见问题

### 忘记密码怎么办？

如果您忘记了管理员密码，可以通过以下步骤在服务器上重置：

1.  通过 SSH 或其他方式登录到您的服务器。

2.  进入您存放 `docker-compose.yml` 的目录。

3.  执行以下命令来重置指定用户的密码。请将 `<username>` 替换为您要重置密码的用户名（例如 `admin`）。

    ```bash
     docker-compose exec danmu-api python -m src.reset_password <username>
    ```

    > **注意**: 如果您没有使用 `docker-compose`，或者您的容器名称不是 `danmu-api`，请使用 `docker exec` 命令：
    > `docker exec <您的容器名称> python -m src.reset_password <username>`

4.  命令执行后，终端会输出一个新的随机密码。请立即使用此密码登录，并在 "设置" -> "账户安全" 页面中修改为您自己的密码。

### 数据库文件越来越大怎么办？

随着时间的推移，数据库占用的磁盘空间可能会逐渐增大。这通常由两个原因造成：

1.  **应用日志**: 任务历史、API访问记录等会存储在数据库中。这些日志会由内置的 **“数据库维护”** 定时任务自动清理（默认保留最近3天）。
2.  **MySQL二进制日志 (Binlog)**: 这是MySQL用于数据恢复和主从复制的日志，如果不进行管理，它会持续增长。

本项目内置的“数据库维护”任务会**尝试自动清理**旧的Binlog文件。但由于权限问题，您可能会在日志中看到“Binlog 清理失败”的警告。这是一个正常且可安全忽略的现象。

如果您关心磁盘空间占用，并希望启用Binlog的自动清理功能，请参阅详细的解决方案：

- **[缓存日志清理任务说明](./缓存日志清理任务说明.md)**

> **对于PostgreSQL用户**: PostgreSQL没有Binlog机制，其WAL日志通常会自动管理，因此空间占用问题没有MySQL那么突出。您只需关注应用日志的自动清理即可。



### 贡献者

<a href="https://github.com/l429609201/misaka_danmu_server/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=l429609201/misaka_danmu_server" alt="contributors" />
</a>

## 参考项目

 - [danmuku](https://github.com/lyz05/danmaku)
 - [emby-toolkit](https://github.com/hbq0405/emby-toolkit)      
 - [swagger-ui](https://github.com/swagger-api/swagger-ui)
 - [imdbsource](https://github.com/wumode/MoviePilot-Plugins/tree/main/plugins.v2/imdbsource)
