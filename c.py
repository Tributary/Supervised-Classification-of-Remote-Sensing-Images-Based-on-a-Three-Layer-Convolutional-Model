import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyArrowPatch

def create_simple_cnn_diagram():
    """创建修复Flatten显示问题的CNN架构图"""
    fig, ax = plt.subplots(figsize=(20, 10))
    # 修复1: 增加x轴范围，确保FLATTEN层能完整显示
    ax.set_xlim(0, 20)  # 从16增加到20 ---------------------------- 这里调整整个画布的宽度
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'input': '#FF6B6B',
        'conv_block': '#4ECDC4',
        'flatten': '#96CEB4',
        'fc': '#FFEAA7',
        'output': '#FF9FF3',
        'conv': '#45B7D1',
        'pool': '#5778FF',
        'text': '#333333'
    }
    
    # 调整模块位置和大小 - 修复2: 调整模块位置，确保都在画布内
    modules = [
        {'name': 'CONV BLOCK 3', 'x': 8.0, 'y': 5, 'width': 3.0, 'height': 5.0,  # 向左移动CONV BLOCK
         'color': colors['conv_block'], 'type': 'conv_block'},
        {'name': 'FLATTEN', 'x': 12.5, 'y': 5, 'width': 3.0, 'height': 2.0,  # 增加宽度，向左移动 ---------------------------- 这里调整FLATTEN层的位置和宽度
         'color': colors['flatten'], 'type': 'flatten'},
        {'name': 'FC LAYERS', 'x': 16.5, 'y': 5, 'width': 3.0, 'height': 3.0,  # 向左移动，增加宽度
         'color': colors['fc'], 'type': 'fc'}
    ]
    
    # 绘制主模块
    for module in modules:
        # 绘制模块外框
        rect = patches.Rectangle(
            (module['x'] - module['width']/2, module['y'] - module['height']/2),
            module['width'], module['height'],
            linewidth=3, edgecolor='black',
            facecolor=module['color'], alpha=0.8
        )
        ax.add_patch(rect)
        
        # 卷积块特殊处理
        if module['type'] == 'conv_block':
            # 在卷积块内绘制子层
            sublayers = [
                'Conv2d (3×3)', 'ReLU', 'BatchNorm', 
                'Conv2d (3×3)', 'ReLU', 'BatchNorm', 
                'MaxPool2d (2×2)', 'Dropout (0.25)'
            ]
            sublayer_colors = [colors['conv'], '#FF9999', '#99FF99', 
                             colors['conv'], '#FF9999', '#99FF99', 
                             colors['pool'], '#CCCCCC']
            
            for i, (sublayer, sub_color) in enumerate(zip(sublayers, sublayer_colors)):
                y_pos = module['y'] + 1.8 - i*0.6
                
                # 子层矩形
                sub_rect = patches.Rectangle(
                    (module['x'] - 1.3, y_pos - 0.25),
                    2.6, 0.5,
                    linewidth=1.5, edgecolor='black',
                    facecolor=sub_color, alpha=0.8
                )
                ax.add_patch(sub_rect)
                
                # 子层名称
                ax.text(module['x'], y_pos, sublayer, 
                       ha='center', va='center', fontsize=10, fontweight='bold', color='black')
        
        # Flatten层特殊处理
        elif module['type'] == 'flatten':
            # 绘制Flatten操作的示意图
            # 左侧：8×8×128的特征图
            ax.text(module['x'] - 1.0, module['y'] + 0.3, '8×8×128', 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='#333333')
            
            # 特征图堆叠
            for c in range(3):
                offset = c * 0.12
                rect = patches.Rectangle(
                    (module['x'] - 1.0 - 0.2 + offset, module['y'] - 0.3 + offset),
                    0.4, 0.6,
                    linewidth=1, edgecolor='black',
                    facecolor=colors['conv'], alpha=0.7 - c*0.2
                )
                ax.add_patch(rect)
            
            # 中间的箭头
            ax.arrow(module['x'] - 0.5, module['y'], 1.0, 0, 
                    head_width=0.12, head_length=0.12, 
                    fc='#666666', ec='#666666', linewidth=2.5)
            
            # 右侧：一维向量表示
            vector_x = module['x'] + 0.8
            ax.text(vector_x, module['y'] + 0.3, '8192', 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='#333333')
            
            # 向量元素
            for i in range(5):
                rect = patches.Rectangle(
                    (vector_x - 0.1, module['y'] - 0.4 + i*0.2),
                    0.2, 0.15,
                    linewidth=1, edgecolor='black',
                    facecolor=colors['flatten'], alpha=0.7
                )
                ax.add_patch(rect)
            
            ax.text(vector_x + 0.3, module['y'], '...', fontsize=16, fontweight='bold')
            
            # 计算公式
            ax.text(module['x'], module['y'] - 0.8, '8 × 8 × 128 = 8192', 
                   ha='center', va='center', fontsize=11, fontweight='bold', color='blue')
        
        # FC LAYERS特殊处理
        elif module['type'] == 'fc':
            # 绘制FC层的维度变换
            ax.text(module['x'], module['y'] + 0.8, 'CLASSIFICATION', 
                   ha='center', va='center', fontsize=13, fontweight='bold', color='blue')
            
            # 维度转换箭头
            dim_x = module['x'] - 1.0
            dim_y = module['y']
            
            # 8192
            ax.text(dim_x, dim_y, '8192', 
                   ha='center', va='center', fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            
            # 箭头1
            ax.arrow(dim_x + 0.3, dim_y, 0.4, 0, 
                    head_width=0.08, head_length=0.08, 
                    fc='#666666', ec='#666666', linewidth=2)
            
            # 512
            ax.text(dim_x + 1.0, dim_y, '512', 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            
            # 箭头2
            ax.arrow(dim_x + 1.3, dim_y, 0.4, 0, 
                    head_width=0.08, head_length=0.08, 
                    fc='#666666', ec='#666666', linewidth=2)
            
            # 21
            ax.text(dim_x + 2.0, dim_y, '21', 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
        
        # 添加模块名称
        name_lines = module['name'].split('\n')
        for i, line in enumerate(name_lines):
            ax.text(module['x'], module['y'] + module['height']/2 - 0.5 - i*0.4, line, 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='black')
        
        # 添加输出形状信息
        if module['name'] == 'CONV BLOCK 3':
            ax.text(module['x'], module['y'] - module['height']/2 + 0.6, 
                   'Input: 16×16×64', ha='center', va='center', fontsize=10, style='italic', color='#333333')
            ax.text(module['x'], module['y'] - module['height']/2 + 0.4, 
                   'Output: 8×8×128', ha='center', va='center', fontsize=12, fontweight='bold', style='italic', color='#333333')
            ax.text(module['x'], module['y'] - module['height']/2 + 0.2, 
                   '128 feature maps', ha='center', va='center', fontsize=11, color='#333333')
        elif module['name'] == 'FLATTEN':
            ax.text(module['x'], module['y'] - module['height']/2 + 0.4, 
                   'Flatten Operation', ha='center', va='center', fontsize=10, style='italic', color='#333333')
        elif module['name'] == 'FC LAYERS':
            ax.text(module['x'], module['y'] - module['height']/2 + 0.4, 
                   '2 Fully Connected Layers', ha='center', va='center', fontsize=11, color='#333333')
    
    # 绘制连接箭头
    for i in range(len(modules)-1):
        x1 = modules[i]['x'] + modules[i]['width']/2
        x2 = modules[i+1]['x'] - modules[i+1]['width']/2
        y = modules[i]['y']
        
        arrow = FancyArrowPatch(
            (x1, y), (x2, y),
            arrowstyle='->', mutation_scale=25,
            linewidth=2.5, color='#666666'
        )
        ax.add_patch(arrow)
    
    # 添加标题
    ax.text(12.0, 9.2, 'CNN Architecture - From Convolution to Classification', 
           ha='center', va='center', fontsize=20, fontweight='bold', color='#333333')
    
    # 添加图例
    legend_elements = [
        patches.Patch(facecolor=colors['conv_block'], edgecolor='black', alpha=0.8, label='Convolutional Block'),
        patches.Patch(facecolor=colors['flatten'], edgecolor='black', alpha=0.8, label='Flatten Layer'),
        patches.Patch(facecolor=colors['fc'], edgecolor='black', alpha=0.8, label='Fully Connected Layers'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
             fontsize=10, framealpha=0.9)
    
    # 添加详细的层类型图例
    detail_legend = [
        patches.Patch(facecolor=colors['conv'], edgecolor='black', alpha=0.8, label='Conv2d (3×3)'),
        patches.Patch(facecolor='#FF9999', edgecolor='black', alpha=0.8, label='ReLU Activation'),
        patches.Patch(facecolor='#99FF99', edgecolor='black', alpha=0.8, label='BatchNorm'),
        patches.Patch(facecolor=colors['pool'], edgecolor='black', alpha=0.8, label='MaxPool2d (2×2)'),
        patches.Patch(facecolor='#CCCCCC', edgecolor='black', alpha=0.8, label='Dropout (0.25)'),
    ]
    
    ax.legend(handles=detail_legend, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
             fontsize=9, framealpha=0.9, title='Operations in Conv Block')
    
    plt.tight_layout()
    plt.savefig('complete_flatten_cnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 完整版CNN架构图已保存为: complete_flatten_cnn_architecture.png")
    return fig

def main():
    """主函数"""
    print("=" * 70)
    print("🎨 生成完整显示Flatten的CNN架构图")
    print("=" * 70)
    
    try:
        # 生成修复版架构图
        print("\n生成完整版架构图...")
        fig = create_simple_cnn_diagram()
        
        print("\n" + "=" * 70)
        print("🎉 完整版CNN架构图生成完成!")
        print("生成的文件: complete_flatten_cnn_architecture.png")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()