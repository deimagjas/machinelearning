# Docusaurus + DevContainer

Este repositorio incluye un entorno de desarrollo basado en DevContainer para lanzar Docusaurus y documentar proyectos de MLX.

## Lanzar Docusaurus en DevContainer

1. Abre el proyecto en VS Code y selecciona "Reopen in Container".
2. El entorno instalará automáticamente Docusaurus y sus dependencias.
3. Para iniciar el servidor de documentación ejecuta:

```bash
cd website
npm run start
```

La documentación de la carpeta `docs/` estará disponible en Docusaurus.

## Seguridad y buenas prácticas
- La imagen base es oficial de Microsoft y actualizada.
- Se incluyen extensiones recomendadas para documentación y formato.
- El puerto 3000 está expuesto únicamente para el servidor de Docusaurus.
- El montaje de la carpeta `docs/` es de solo lectura para el contenedor.

## Estructura relevante
- `.devcontainer/` : Configuración del entorno de desarrollo.
- `docs/` : Documentación en formato Markdown.
- `website/` : Proyecto Docusaurus generado automáticamente.

---

Para más información sobre Docusaurus visita: https://docusaurus.io/docs
