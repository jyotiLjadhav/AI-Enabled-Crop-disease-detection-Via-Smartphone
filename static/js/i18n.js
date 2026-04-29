(function () {
  const DICT = {
    en: {
      brand: "Crop Disease AI Detection System",
      nav_home: "Home",
      nav_history: "History",
      app_title: "AI Crop Disease Detection",
      app_subtitle: "Smart Deep Learning System for Instant Plant Disease Prediction",
    },
    hi: {
      brand: "फसल रोग एआई पहचान प्रणाली",
      nav_home: "होम",
      nav_history: "इतिहास",
      app_title: "एआई फसल रोग पहचान",
      app_subtitle: "तुरंत पौध रोग पहचान के लिए स्मार्ट डीप लर्निंग सिस्टम",
    },
    mr: {
      brand: "पीक रोग एआय ओळख प्रणाली",
      nav_home: "मुख्यपृष्ठ",
      nav_history: "इतिहास",
      app_title: "एआय पीक रोग ओळख",
      app_subtitle: "तत्काळ रोग भाकितासाठी स्मार्ट डीप लर्निंग सिस्टम",
    },
    kn: {
      brand: "ಬೆಳೆ ರೋಗ ಎಐ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ",
      nav_home: "ಮುಖಪುಟ",
      nav_history: "ಇತಿಹಾಸ",
      app_title: "ಎಐ ಬೆಳೆ ರೋಗ ಪತ್ತೆ",
      app_subtitle: "ತಕ್ಷಣದ ಸಸ್ಯ ರೋಗ ಪತ್ತೆಗಾಗಿ ಸ್ಮಾರ್ಟ್ ಡೀಪ್ ಲರ್ನಿಂಗ್ ವ್ಯವಸ್ಥೆ",
    },
  };

  function getLang() {
    const stored = localStorage.getItem("lang");
    if (stored && DICT[stored]) return stored;
    return "en";
  }

  function t(key) {
    const lang = getLang();
    return (DICT[lang] && DICT[lang][key]) || (DICT.en[key] || key);
  }

  function applyTranslations() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (!key) return;
      el.textContent = t(key);
    });

    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      const key = el.getAttribute("data-i18n-placeholder");
      if (!key) return;
      el.setAttribute("placeholder", t(key));
    });
  }

  window.I18N = {
    DICT,
    getLang,
    setLang: function (lang) {
      if (!DICT[lang]) lang = "en";
      localStorage.setItem("lang", lang);
      applyTranslations();
    },
    t,
    applyTranslations,
  };
})();

