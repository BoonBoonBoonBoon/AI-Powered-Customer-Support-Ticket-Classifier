from app.models.runtime_loader import ensure_model_ready

def main():
    loaded = ensure_model_ready('staging')
    print('manifest model_id:', loaded['manifest'].get('model_id'))
    print('artifacts:', {k: str(v) for k, v in loaded['artifacts'].items()})
    print('OK')

if __name__ == '__main__':
    main()
