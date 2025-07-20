#!/usr/bin/env python3
"""
Скрипт для проверки настроек .gitignore
"""

import os
import subprocess
import sys

def run_command(cmd):
    """Выполняет команду и возвращает результат"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)

def check_gitignore():
    """Проверяет настройки .gitignore"""
    print("🔍 Проверка настроек .gitignore")
    print("=" * 50)
    
    # Список файлов для проверки
    test_files = [
        "deep_sort_pytorch/configs/deep_sort.yaml",
        "deep_sort_pytorch/deep_sort/deep/checkpoint/resnet18-5c106cde.pth",
        "deep_sort_pytorch/detector/YOLOv3/weight/yolov3.weights",
        "deep_sort_pytorch/detector/YOLOv5/yolov5s.pt",
        "output/results.avi",
        "*.log",
        "temp/",
        "results/"
    ]
    
    print("📋 Проверяемые файлы и папки:")
    for file_path in test_files:
        success, output = run_command(f'git check-ignore "{file_path}"')
        if success and output:
            print(f"✅ {file_path} - ИГНОРИРУЕТСЯ")
        elif success:
            print(f"❌ {file_path} - НЕ игнорируется")
        else:
            print(f"⚠️ {file_path} - Ошибка проверки: {output}")
    
    print("\n" + "=" * 50)
    
    # Проверяем статус Git
    print("📊 Статус Git:")
    success, output = run_command("git status --porcelain")
    if success:
        if output:
            print("📝 Изменения в репозитории:")
            for line in output.split('\n'):
                if line.strip():
                    print(f"   {line}")
        else:
            print("✅ Репозиторий чист")
    else:
        print(f"❌ Ошибка получения статуса: {output}")

def show_gitignore_rules():
    """Показывает правила .gitignore"""
    print("\n📄 Правила .gitignore:")
    print("=" * 50)
    
    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    print(f"{i:3d}: {line}")
                elif line.startswith('#'):
                    print(f"{i:3d}: {line}")
    else:
        print("❌ Файл .gitignore не найден")

def main():
    """Главная функция"""
    print("🚀 Проверка настроек Git для DeepSORT")
    print("=" * 60)
    
    # Проверяем, что мы в Git репозитории
    if not os.path.exists('.git'):
        print("❌ Это не Git репозиторий")
        return
    
    check_gitignore()
    show_gitignore_rules()
    
    print("\n" + "=" * 60)
    print("💡 Рекомендации:")
    print("1. Все файлы DeepSORT должны игнорироваться")
    print("2. Веса моделей (*.pt, *.pth, *.weights) не должны попадать в Git")
    print("3. Конфигурационные файлы (*.yaml) должны игнорироваться")
    print("4. Временные файлы и результаты должны игнорироваться")
    
    print("\n📚 Дополнительная информация:")
    print("- См. файл DEEPSORT_SETUP.md для инструкций по настройке")
    print("- Пример конфигурации: deep_sort_pytorch/configs/deep_sort_example.yaml")

if __name__ == "__main__":
    main() 