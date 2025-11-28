"""Script independiente para generar informe consolidado de modelos entrenados.

Este script puede ejecutarse directamente sin necesidad de Kedro:
    python scripts/generate_models_report.py

O puede ejecutarse a través de Kedro:
    kedro run --pipeline=reporting
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Importar directamente la función sin pasar por __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "reporting_nodes",
    project_root / "src" / "proyecto_ml_clashroyale" / "pipelines" / "nodes" / "reporting_nodes.py"
)
reporting_nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting_nodes)
generate_models_report = reporting_nodes.generate_models_report


def load_json_file(filepath: Path) -> dict:
    """Cargar archivo JSON si existe.
    
    Args:
        filepath: Ruta al archivo JSON
        
    Returns:
        Diccionario con los datos o None si no existe
    """
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Error al cargar {filepath}: {e}")
            return None
    return None


def main():
    """Función principal para generar el informe."""
    print("=" * 70)
    print("📊 GENERADOR DE INFORME CONSOLIDADO DE MODELOS")
    print("=" * 70)
    
    # Rutas a los archivos de resultados
    reporting_dir = project_root / "data" / "08_reporting"
    model_output_dir = project_root / "data" / "07_model_output"
    
    print(f"\n📁 Directorio de reportes: {reporting_dir}")
    print(f"📁 Directorio de métricas: {model_output_dir}\n")
    
    # Cargar todos los resultados disponibles
    print("📥 Cargando resultados de modelos...")
    
    classification_comparison = load_json_file(reporting_dir / "classification_comparison.json")
    regression_comparison = load_json_file(reporting_dir / "regression_comparison.json")
    clustering_comparison = load_json_file(reporting_dir / "clustering_comparison.json")
    pca_metrics = load_json_file(model_output_dir / "pca_metrics.json")
    umap_metrics = load_json_file(model_output_dir / "umap_metrics.json")
    anomaly_detection_comparison = load_json_file(reporting_dir / "anomaly_detection_comparison.json")
    association_rules_comparison = load_json_file(reporting_dir / "association_rules_comparison.json")
    classification_shap_summary = load_json_file(reporting_dir / "classification_shap_summary.json")
    regression_shap_summary = load_json_file(reporting_dir / "regression_shap_summary.json")
    
    # Mostrar qué archivos se cargaron
    loaded_files = []
    if classification_comparison:
        loaded_files.append("✅ Clasificación")
    if regression_comparison:
        loaded_files.append("✅ Regresión")
    if clustering_comparison:
        loaded_files.append("✅ Clustering")
    if pca_metrics:
        loaded_files.append("✅ PCA")
    if umap_metrics:
        loaded_files.append("✅ UMAP")
    if anomaly_detection_comparison:
        loaded_files.append("✅ Detección de Anomalías")
    if association_rules_comparison:
        loaded_files.append("✅ Reglas de Asociación")
    if classification_shap_summary:
        loaded_files.append("✅ SHAP Clasificación")
    if regression_shap_summary:
        loaded_files.append("✅ SHAP Regresión")
    
    print(f"\n📋 Archivos cargados ({len(loaded_files)}):")
    for file in loaded_files:
        print(f"   {file}")
    
    if not any([
        classification_comparison, regression_comparison, clustering_comparison,
        pca_metrics, anomaly_detection_comparison, association_rules_comparison
    ]):
        print("\n❌ Error: No se encontraron archivos de resultados.")
        print("💡 Ejecuta primero los pipelines de entrenamiento:")
        print("   - kedro run --pipeline=classification")
        print("   - kedro run --pipeline=regression")
        print("   - kedro run --pipeline=unsupervised_learning")
        print("   - kedro run --pipeline=anomaly_detection")
        print("   - kedro run --pipeline=association_rules")
        return 1
    
    # Generar el informe
    print("\n🔄 Generando informe consolidado...")
    
    try:
        report = generate_models_report(
            classification_comparison=classification_comparison,
            regression_comparison=regression_comparison,
            clustering_comparison=clustering_comparison,
            pca_metrics=pca_metrics,
            umap_metrics=umap_metrics,
            anomaly_detection_comparison=anomaly_detection_comparison,
            association_rules_comparison=association_rules_comparison,
            classification_shap_summary=classification_shap_summary,
            regression_shap_summary=regression_shap_summary,
        )
        
        # Guardar el informe
        output_file = reporting_dir / "models_report.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Informe generado exitosamente!")
        print(f"📄 Archivo: {output_file}")
        print(f"\n📊 Resumen:")
        print(f"   - Total de modelos: {report['summary']['total_models']}")
        print(f"   - Modelos supervisados: {report['summary']['supervised_models']}")
        print(f"   - Modelos no supervisados: {report['summary']['unsupervised_models']}")
        
        if report['summary']['best_models']:
            print(f"\n🏆 Mejores Modelos:")
            for task, best in report['summary']['best_models'].items():
                print(f"   - {task}: {best.get('model', best.get('algorithm', 'N/A'))}")
        
        print("\n" + "=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error al generar el informe: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

