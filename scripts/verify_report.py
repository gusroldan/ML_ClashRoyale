"""Script para verificar el contenido del informe JSON."""

import json
from pathlib import Path

report_file = Path("data/08_reporting/models_report.json")

if report_file.exists():
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    print("=" * 70)
    print("📊 VERIFICACIÓN DEL INFORME JSON")
    print("=" * 70)
    
    print(f"\n✅ Metadata:")
    print(f"  - Proyecto: {report['metadata']['project_name']}")
    print(f"  - Tipo: {report['metadata']['report_type']}")
    print(f"  - Fecha: {report['metadata']['generation_date']}")
    print(f"  - Versión: {report['metadata']['version']}")
    
    print(f"\n📈 Summary:")
    print(f"  - Total modelos: {report['summary']['total_models']}")
    print(f"  - Modelos supervisados: {report['summary']['supervised_models']}")
    print(f"  - Modelos no supervisados: {report['summary']['unsupervised_models']}")
    
    print(f"\n🏆 Mejores Modelos:")
    for task, best in report['summary']['best_models'].items():
        model_name = best.get('model', best.get('algorithm', 'N/A'))
        print(f"  - {task}: {model_name}")
        if 'roc_auc' in best:
            print(f"    ROC-AUC: {best['roc_auc']:.4f}")
        if 'mae' in best:
            print(f"    MAE: {best['mae']:.4f}")
        if 'silhouette_score' in best:
            print(f"    Silhouette: {best['silhouette_score']:.4f}")
    
    print(f"\n📋 Contenido del Informe:")
    print(f"  - Clasificación: {len(report['supervised_learning']['classification']['models'])} modelos")
    print(f"  - Regresión: {len(report['supervised_learning']['regression']['models'])} modelos")
    print(f"  - Clustering: {report['unsupervised_learning']['clustering']['total_algorithms']} algoritmos")
    
    if 'pca' in report['unsupervised_learning']['dimensionality_reduction']:
        print(f"  - PCA: ✅ Incluido")
    if 'umap' in report['unsupervised_learning']['dimensionality_reduction']:
        print(f"  - UMAP: ✅ Incluido")
    if 'anomaly_detection' in report['unsupervised_learning']:
        print(f"  - Detección de Anomalías: ✅ Incluido")
    if 'association_rules' in report['unsupervised_learning']:
        print(f"  - Reglas de Asociación: ✅ Incluido")
    
    if report['interpretability']['shap_analysis']:
        print(f"  - SHAP Analysis: ✅ Incluido")
    
    print(f"\n✅ Informe JSON válido y completo!")
    print(f"📄 Ubicación: {report_file.absolute()}")
    print("=" * 70)
else:
    print(f"❌ Archivo no encontrado: {report_file}")

