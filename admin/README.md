# AI Video Pilot — Admin Viewer

Простая локальная админка на TypeScript (Vite + React): выбор видео и JSON трекинга, просмотр с наложением bbox / ID / траекторий.

## Запуск

```bash
cd admin
npm install
npm run dev
```

Откройте URL из терминала (обычно http://localhost:5173).

## Как пользоваться

1. В списке **Из data/video** выберите ролик — JSON `data/results/{имя}/tracking.json` подставится сам
2. Тумблеры подписей / поз / траекторий запоминаются в `localStorage`

Видео отдаёт Vite из `../data/video`, JSON и кропы — из `../data/results`.
По умолчанию сервер слушает только `localhost` (без auth). Для LAN в `vite.config.ts` поставьте `host: true`.
