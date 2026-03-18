# 环境变量配置指南

## HF_TOKEN 配置方式

HF_TOKEN 用于访问 Hugging Face 的 pyannote 说话人分离模型。支持以下三种配置方式（按优先级排序）：

### 方式 1：在 config.yaml 中配置（推荐）

编辑 `config.yaml`：

```yaml
diarization:
  enabled: true
  huggingface_token: "hf_YOUR_TOKEN_HERE"  # 直接填写 token
  # 或者使用环境变量引用
  # huggingface_token: "${HF_TOKEN}"  # 从环境变量读取
```

### 方式 2：使用 .env 文件（推荐用于开发环境）

1. 复制示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件：
```bash
HF_TOKEN=hf_YOUR_TOKEN_HERE
```

代码会自动读取项目目录下的 `.env` 文件。

### 方式 3：系统环境变量（推荐用于生产环境）

#### 临时设置（当前会话有效）
```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

#### 永久设置（用户级别）
编辑 `~/.bashrc` 或 `~/.profile`：
```bash
echo 'export HF_TOKEN=hf_YOUR_TOKEN_HERE' >> ~/.bashrc
source ~/.bashrc
```

#### 系统级别（所有用户）
编辑 `/etc/environment`：
```bash
sudo nano /etc/environment
# 添加：HF_TOKEN=hf_YOUR_TOKEN_HERE
```

#### Systemd Service 中设置

编辑 `/etc/systemd/system/whisperlm.service`：

```ini
[Service]
# 方式1：直接设置（不推荐，安全性较低）
Environment="HF_TOKEN=hf_YOUR_TOKEN_HERE"

# 方式2：从系统环境变量读取（推荐）
# 确保在 /etc/environment 中设置了 HF_TOKEN
```

然后重新加载并重启服务：
```bash
sudo systemctl daemon-reload
sudo systemctl restart whisperlm.service
```

## 配置优先级

配置的优先级顺序（高优先级会覆盖低优先级）：

1. **config.yaml** 中的 `huggingface_token` 字段（最高优先级）
2. **.env** 文件中的 `HF_TOKEN`
3. **系统环境变量** `HF_TOKEN`（最低优先级）

## 验证配置

启动服务后，检查日志确认 token 是否加载成功：

```bash
# 如果使用 systemd
sudo journalctl -u whisperlm.service -f | grep -i token

# 如果直接运行
python -m whisperlm.main
# 应该看到：Diarization model loaded（而不是警告）
```

## 获取 HF_TOKEN

1. 访问 https://huggingface.co/settings/tokens
2. 创建新的 Access Token（需要 Read 权限）
3. 申请访问权限：https://huggingface.co/pyannote/speaker-diarization-3.1
4. 复制生成的 token（格式：`hf_xxxxxxxxxxxxxxxxxxxxx`）

## 安全建议

1. **不要**将 token 提交到 Git 仓库
2. **不要**在代码中硬编码 token
3. 使用 `.env` 文件时，确保 `.env` 在 `.gitignore` 中
4. 生产环境推荐使用系统环境变量或 systemd service 配置

