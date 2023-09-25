## API 简介
### 快速开始
1. 启动 server 服务，在项目根目录下执行
```bash
python backend/server.py
```
2. 测试 txt2img 功能
```bash
python backend/agentes/txt2img.py
```

### 目前 API 提供的接口
<table width="100%">
<tr>
<td width="25%" style="text-align: center">功能</td>
<td width="75%" style="text-align: center">参数</td>
</tr>
<tr>
<td width="25%" style="text-align: center">txt2img</td>
<td width="75%" style="text-align: center"> data = {"prompt": "a pretty puppy", 
                                                    "negative_prompt": "ugly, low quaility, blur",
                                                    "seed": 2023} 
                                                    </td>
</tr>
<tr>
<td width="25%" style="text-align: center">lora2img</td>
<td width="75%" style="text-align: center">data = {"prompt": "a pretty puppy", 
                                                    "negative_prompt": "ugly, low quaility, blur",
                                                    "seed": 2023} 
                                                    </td>
</tr>
</table>

## FAQ
1. 模型地址可以在 ```backend/server.py``` 里修改
2. 端口号在 ```backend/server.py``` 修改，建议与 docker 建立容器时一致。
3. 后续会持续增加接口功能。