"""快速单尺寸测试，提取关键指标"""
from ultralytics import YOLO

def test_size(imgsz):
    print(f'\n📏 尺寸 {imgsz}×{imgsz} 验证')
    print('-' * 50)
    model = YOLO('runs/detect/train19/weights/best.pt')
    results = model.val(
        data='datasets/rgbt_subset.yaml', 
        imgsz=imgsz, 
        batch=4, 
        workers=0, 
        plots=False, 
        verbose=False
    )
    p = float(results.box.p)
    r = float(results.box.r)
    map50 = float(results.box.map50)
    map_val = float(results.box.map)
    
    print(f'Precision: {p:.4f}')
    print(f'Recall:    {r:.4f}')
    print(f'mAP50:     {map50:.4f}')
    print(f'mAP50-95:  {map_val:.4f}')
    return p, r, map50, map_val

if __name__ == '__main__':
    print('='*70)
    print('快速尺寸对比测试 (256 vs 512)')
    print('='*70)
    
    p256, r256, m256, m95_256 = test_size(256)
    p512, r512, m512, m95_512 = test_size(512)
    
    print('\n' + '='*70)
    print('📊 对比总结')
    print('='*70)
    print(f'\n{"尺寸":<10} {"Precision":<12} {"Recall":<12} {"mAP50":<12} {"mAP50-95":<12}')
    print('-'*70)
    print(f'{"256":<10} {p256:<12.4f} {r256:<12.4f} {m256:<12.4f} {m95_256:<12.4f}')
    print(f'{"512":<10} {p512:<12.4f} {r512:<12.4f} {m512:<12.4f} {m95_512:<12.4f}')
    print('-'*70)
    
    r_delta = r512 - r256
    print(f'\n💡 Recall 变化: {r_delta:+.4f}')
    if r_delta > 0.05:
        print('   ✓ 明显改善！建议升级到 512')
    elif r_delta > 0.01:
        print('   ✓ 略有改善，可继续测试更大尺寸')
    else:
        print('   ✗ 改善不明显，低 Recall 原因另有其他')
    print('='*70)
