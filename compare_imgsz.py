"""使用 train28 权重快速对比不同 imgsz 的 Recall 变化"""
from ultralytics import YOLO
import sys

def test_imgsz(model_path, imgsz):
    """测试单一 imgsz"""
    print(f'\n📏 imgsz = {imgsz}×{imgsz}')
    print('-' * 50)
    try:
        model = YOLO(model_path)
        results = model.val(
            data='datasets/rgbt_subset.yaml',
            imgsz=imgsz,
            batch=4,
            workers=0,
            plots=False,
            verbose=False,
            device=0
        )
        p = float(results.box.p)
        r = float(results.box.r)
        m50 = float(results.box.map50)
        m95 = float(results.box.map)
        print(f'✓ Precision: {p:.4f} | Recall: {r:.4f} | mAP50: {m50:.4f} | mAP50-95: {m95:.4f}')
        return p, r, m50, m95
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'⚠ OOM (显存不足)')
            return None, None, None, None
        raise

if __name__ == '__main__':
    model_path = 'runs/detect/train28/weights/best.pt'
    
    print('='*70)
    print('快速 imgsz 对比 (train28 权重)')
    print('='*70)
    
    results = {}
    for imgsz in [256, 384, 512]:
        p, r, m50, m95 = test_imgsz(model_path, imgsz)
        if r is not None:
            results[imgsz] = {'p': p, 'r': r, 'm50': m50, 'm95': m95}
    
    if results:
        print('\n' + '='*70)
        print('📊 对比总结')
        print('='*70)
        print(f"\n{'imgsz':<10} {'Precision':<12} {'Recall':<12} {'mAP50':<12}")
        print('-'*70)
        for imgsz in sorted(results.keys()):
            r = results[imgsz]
            print(f"{imgsz:<10} {r['p']:<12.4f} {r['r']:<12.4f} {r['m50']:<12.4f}")
        
        if len(results) >= 2:
            imgsz_list = sorted(results.keys())
            r_delta = results[imgsz_list[-1]]['r'] - results[imgsz_list[0]]['r']
            print(f"\n💡 从 {imgsz_list[0]} 到 {imgsz_list[-1]}, Recall 变化: {r_delta:+.4f}")
            
            if r_delta > 0.05:
                print(f"   ✓ 明显改善！建议升级到 {imgsz_list[-1]}")
            elif r_delta > 0.01:
                print(f"   ✓ 略有改善，可再测试更大尺寸")
            else:
                print("   ✗ 改善不明显，低 Recall 非分辨率问题")
        
        print('='*70)
    else:
        print('❌ 所有测试失败')
        sys.exit(1)
