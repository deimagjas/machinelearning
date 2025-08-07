module.exports = {
  title: 'MLX Examples',
  tagline: 'Documentación de ejemplos MLX',
  url: 'http://localhost:3000',
  baseUrl: '/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'deimagjas',
  projectName: 'machinelearning',
  presets: [
    [
      'classic',
      {
        docs: {
          path: '/workspaces/machinelearning/docs',
          routeBasePath: 'docs',
          sidebarPath: require.resolve('/workspaces/machinelearning/sidebars.js'),
        },
        blog: false,
        theme: {
          customCss: require.resolve('/workspaces/machinelearning/website/src/css/custom.css'),
        },
      },
    ],
  ],
};
