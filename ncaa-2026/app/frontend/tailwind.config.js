/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'ncaa-blue': '#003087',
        'ncaa-gold': '#FFB81C',
      },
    },
  },
  plugins: [],
}
