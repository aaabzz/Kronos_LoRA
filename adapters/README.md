# LoRA推理模型目录

此目录只保存部署到实时预测服务的LoRA Adapter，与训练过程产生的
`checkpoints/` 完全分开。前端不会直接加载训练检查点。

每个子目录代表一个可选择模型，并且只能包含一份Adapter权重。例如：

```text
adapters/
├── 沪深300-LoRA/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── 科技股-LoRA/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

也支持使用 `adapter_model.bin`，但推荐Safetensors格式。同一目录不要同时放
`.safetensors` 和 `.bin`，否则该目录不会加入可用模型列表。服务启动时扫描
所有完整子目录；模型名称取目录名，权重在首次选择时加载并缓存。

基础模型和Tokenizer仍从以下目录加载：

```text
pretrained/Kronos-base/
pretrained/Kronos-Tokenizer-base/
```

文件完整后重启FastAPI，前端“LoRA微调模型”选项会自动变为可用。

如需加载其他部署目录，可设置：

```powershell
$env:KRONOS_LORA_ADAPTER_PATH="D:\models\my-kronos-lora"
```

不要把训练中尚未完成写入的Adapter目录直接配置为推理目录。
